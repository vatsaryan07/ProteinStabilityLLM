# Protein Stability Prediction Project

## Introduction

### Purpose
Protein engineering is a dynamic field focusing on creating synthetic proteins with desired functions, such as catalyzing chemical reactions or treating diseases. A significant challenge in this domain is the instability of many proteins, leading to potential loss of functionality. Predicting protein stability before synthesis can streamline the protein design process, saving time and resources. In this project, our goal is to leverage large language protein models to predict protein stability using amino acid sequences.

### Dataset
The TAPE dataset, curated by Rao et al. in 2019, comprises 68,977 protein sequences with corresponding stability values. These sequences consist of amino acids represented as (X1, X2, …, XL), with X being one of 25 characters (20 standard amino acids, 2 non-standard amino acids, 2 ambiguous amino acids, and 1 for unknown amino acids).

### Contribution
While Rao et al. experimented with Transformer, LSTM, and ResNet, we aim to explore the performance of recently published protein language models (UniRep, ESM2, ProtBert, ProtTrans). These models were pre-trained on large protein sequence databases, and our objective is to assess their impact on predicting protein stability scores.

## Approach

### Dataset Split
The TAPE dataset was split into three subsets by the authors, with 53,614 samples in the training dataset, 2,512 samples in the validation dataset, and 12,851 samples in the test dataset. We trained our models with the training dataset, fine-tuned the hyperparameters with the validation dataset, and evaluated their performance on the held-out test dataset. Finally, we evaluated the performance of the three subsets using MSE, RMSE, R2, and Spearman ρ of the predicted stability scores and the true stability scores. We also compared the results of Spearman ρ with the results from Rao et al.[2] on the test dataset. They implemented an one-hot encoding model, a LSTM model, a transformer model, and a ResNet model that were pre-trained on the TAPE dataset using masked-token prediction or next-token prediction.

### Model Fine-Tuning Details

#### ESM2
ESM2, Evolutionary Scale Modeling, is a deep contextual language model trained on 250 million protein sequences using unsupervised learning. It consists of 34-layer Transformer models with ~113 M parameters. We fine-tuned the model using Adam optimizer with a learning rate of 0.0001, a batch size of 64, and trained for 10 epochs with Mean Square Error as the loss function.

#### UniRep
UniRep is a multilayer long short-term memory (mLSTM) "babbler" deep representation learner for protein engineering informatics. We experimented with three architecture variants of 64 units and fine-tuned UniRep on the TAPE dataset using the Adam optimizer with a learning rate of 0.001, a batch size of 512, and up to 10 epochs with early stopping based on validation loss. Dropout regularization with a rate of 0.1 was applied.

#### ProtBert
ProteinBERT is a protein language model pre-trained on the UniRef90 dataset on approximately 106M proteins. We fine-tuned the model by encoding the TAPE dataset into ProteinBERT compatible data through the ProteinBERT library. The Adam optimizer with a learning rate of 0.0001 was used, and the model was trained on the TAPE train dataset by freezing all but the last two layers.

#### ProtTrans
ProtTrans is a deep learning architecture designed to model protein sequences, pre-trained on a larger dataset. To fine-tune ProtTrans for the stability score prediction task using the TAPE dataset, Adam optimizer was used with an Mean Squared Error (MSE) loss function. The training was done with a learning rate of 0.0001, batch size of 64, and for one epoch.

## Performance

This section provides a comparative analysis of the performance of all the four pretrained models on the stability score prediction task for the train (Table 1), validation (Table 2), and test (Table 3) data sets. The metrics chosen are R-squared, Root Mean Square Error (RMSE), Mean Square Error (MSE), and Spearman ρ to measure how close the predicted stability scores are to the ground truth values.

### Table 1: Model performance on the train dataset

|R2    |RMSE |MSE  |Spearman ρ|
|------|-----|-----|-----------|
|UniRep|0.2  |0.39 |0.42       |
|ESM   |0.63 |0.35 |0.86       |
|ProtBert|0.28|0.48 |0.43       |
|ProtTrans|0.47|0.34|0.77       |

### Table 2: Model performance on the validation dataset

|R2    |RMSE |MSE  |Spearman ρ|
|------|-----|-----|-----------|
|UniRep|0.12 |0.4  |0.39       |
|ESM   |0.53 |0.45 |0.75       |
|ProtBert|0.38|0.51 |0.60       |
|ProtTrans|0.26|0.40|0.73       |

### Table 3: Model performance on the test dataset

|R2    |RMSE |MSE  |Spearman ρ|
|------|-----|-----|-----------|
|One-hot from Rao et al[2]|N/A |N/A |0.19|
|Transformer from Rao et al[2]|N/A |N/A |0.73|
|LSTM from Rao et al[2]|N/A |N/A |0.69|
|ResNet from Rao et al[2]|N/A |N/A |0.73|
|UniRep|-0.37|0.4  |0.32       |
|ESM   |-0.50|0.50 |0.70       |
|ProtBert|0.04|0.43 |0.60       |
|ProtTrans|-0.35|0.44|0.71       |

From the results in the test dataset, we observed that among the four pre-trained models, ESM and ProtTrans performed better than the other two models. These two models were able to achieve ~0.7 Spearman ρ in the test set. However, these models did not achieve better results compared to the Transformer and ResNet models that achieved ~0.73 Spearman ρ in Rao’s et al publication. The reason may be coming from the fact that those models in Rao’s et al were pre-trained directly on the TAPE dataset, whereas the four large protein language models we tested were trained on protein datasets that include all possible proteins in general. Future work on a more extensive hyperparameter tuning and longer training epochs of those protein language models may be helpful to provide a better comparison of the models’ performance.
