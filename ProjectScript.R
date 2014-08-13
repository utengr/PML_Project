library(caret)
library(randomForest)
dat <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
set.seed(12345)
inTrain <- createDataPartition(y = dat$classe, p = 0.6, list = FALSE)
datTrain <- dat[inTrain,]
datNotTrain <- dat[-inTrain,]
inTest <- createDataPartition( y=datNotTrain$classe, p = 0.5, list = FALSE)
datTest <- datNotTrain[inTest,]
datValidate <- datNotTrain[-inTest,]
datTrain.clean <- datTrain[,apply(!is.na(datTrain), 2, all)]
datTrain.clean <- datTrain.clean[,-c(1,2,3,4,5,6,7,12,13,14,15,
                                     16,17,18,19,20,43,44,45,46,
                                     47,48,52,53,54,55,56,57,58,
                                     59,60,74,75,76,77,78,79,80,
                                     81,82)]
datTrain.clean$classe <- as.factor(datTrain.clean$classe)
ctrl <- trainControl(method = "cv", number = 10)
grid_rf <- expand.grid(.mtry = c(2,4,8,16,52))
set.seed(12345)
model.train <- train(classe ~., data = datTrain.clean, method = "rf",
                     trControl = ctrl,
                     tuneGrid = grid_rf,
                     importance = TRUE,
                     ntree = 100)

model <- model.train$finalModel
model
# importance(model)
varImpPlot(model, n.var=52)

plot(model)
legend("topright", lwd = 1, col = c(1,2,3,4,5,6),
       legend = c("OOB", "A", "B", "C", "D", "E"))

datValidate.clean <- datValidate[,apply(!is.na(datValidate), 2, all)]
datValidate.clean <- datValidate.clean[,-c(1,2,3,4,5,6,7,12,13,14,15,
                                     16,17,18,19,20,43,44,45,46,
                                     47,48,52,53,54,55,56,57,58,
                                     59,60,74,75,76,77,78,79,80,
                                     81,82)]
datValidate.clean$classe <- as.factor(datValidate.clean$classe)

datTest.clean <- datTest[,apply(!is.na(datTest), 2, all)]
datTest.clean <- datTest.clean[,-c(1,2,3,4,5,6,7,12,13,14,15,
                                           16,17,18,19,20,43,44,45,46,
                                           47,48,52,53,54,55,56,57,58,
                                           59,60,74,75,76,77,78,79,80,
                                           81,82)]
datTest.clean$classe <- as.factor(datTest.clean$classe)

val.pred <- predict(model.train, datValidate.clean)
confusionMatrix(val.pred, datValidate$classe)

val.test <- predict(model.train, datTest.clean)
confusionMatrix(val.test, datTest$classe)

testproj <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)
testproj.clean <- testproj[,apply(!is.na(testproj), 2, all)]
testproj.clean <- testproj.clean[,-c(1,2,3,4,5,6,7,12,13,14,15,
                                   16,17,18,19,20,43,44,45,46,
                                   47,48,52,53,54,55,56,57,58,
                                   59,60,74,75,76,77,78,79,80,
                                   81,82)]

test.pred <- predict(model.train, testproj.clean[,1:59])
test.pred <- as.character(test.pred)

pml_write_files(test.pred)