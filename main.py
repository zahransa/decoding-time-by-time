
import mne
import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import SlidingEstimator, cross_val_multiscore
from pathlib import Path
import json
#to fix
#qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load inputs from config.json
with open('config.json') as config_json:
    config = json.load(config_json)

# == LOAD DATA ==
fname = config['epo']
epochs = mne.read_epochs(fname)


epochs_auditory = epochs['auditory']




# First, create X and y.
epochs_auditory_grad = epochs_auditory.copy().pick_types(meg='grad')
X = epochs_auditory_grad.get_data()
y = epochs_auditory_grad.events[:, 2]

# Classifier pipeline. No need for vectorization as in the previous example.
clf = make_pipeline(StandardScaler(),
                    LogisticRegression())

# The "sliding estimator" will train the classifier at each time point.
scoring = 'roc_auc'
time_decoder = SlidingEstimator(clf, scoring=scoring, n_jobs=1, verbose=True)

# Run cross-validation.
n_splits = 5
scores = cross_val_multiscore(time_decoder, X, y, cv=5, n_jobs=1)

# Mean scores across cross-validation splits, for each time point.
mean_scores = np.mean(scores, axis=0)

# Mean score across all time points.
mean_across_all_times = round(np.mean(scores), 3)
print(f'\n=> Mean CV score across all time points: {mean_across_all_times:.3f}')



fig, ax = plt.subplots()

ax.axhline(0.5, color='k', linestyle='--', label='chance')  # AUC = 0.5
ax.axvline(0, color='k', linestyle='-')  # Mark time point zero.
ax.plot(epochs.times, mean_scores, label='score')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean ROC AUC')
ax.legend()
ax.set_title('Left vs Right')
fig.suptitle('Sensor Space Decoding')
fig.savefig(os.path.join('out_figs', 'Sensor-Space-Decoding.png'))


