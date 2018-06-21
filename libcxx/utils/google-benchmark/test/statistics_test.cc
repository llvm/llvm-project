//===---------------------------------------------------------------------===//
// statistics_test - Unit tests for src/statistics.cc
//===---------------------------------------------------------------------===//

#include "../src/statistics.h"
#include "gtest/gtest.h"

namespace {
TEST(StatisticsTest, Mean) {
  std::vector<double> Inputs;
  {
    Inputs = {42, 42, 42, 42};
    double Res = benchmark::StatisticsMean(Inputs);
    EXPECT_DOUBLE_EQ(Res, 42.0);
  }
  {
    Inputs = {1, 2, 3, 4};
    double Res = benchmark::StatisticsMean(Inputs);
    EXPECT_DOUBLE_EQ(Res, 2.5);
  }
  {
    Inputs = {1, 2, 5, 10, 10, 14};
    double Res = benchmark::StatisticsMean(Inputs);
    EXPECT_DOUBLE_EQ(Res, 7.0);
  }
}

TEST(StatisticsTest, Median) {
  std::vector<double> Inputs;
  {
    Inputs = {42, 42, 42, 42};
    double Res = benchmark::StatisticsMedian(Inputs);
    EXPECT_DOUBLE_EQ(Res, 42.0);
  }
  {
    Inputs = {1, 2, 3, 4};
    double Res = benchmark::StatisticsMedian(Inputs);
    EXPECT_DOUBLE_EQ(Res, 2.5);
  }
  {
    Inputs = {1, 2, 5, 10, 10};
    double Res = benchmark::StatisticsMedian(Inputs);
    EXPECT_DOUBLE_EQ(Res, 5.0);
  }
}

TEST(StatisticsTest, StdDev) {
  std::vector<double> Inputs;
  {
    Inputs = {101, 101, 101, 101};
    double Res = benchmark::StatisticsStdDev(Inputs);
    EXPECT_DOUBLE_EQ(Res, 0.0);
  }
  {
    Inputs = {1, 2, 3};
    double Res = benchmark::StatisticsStdDev(Inputs);
    EXPECT_DOUBLE_EQ(Res, 1.0);
  }
}

}  // end namespace
