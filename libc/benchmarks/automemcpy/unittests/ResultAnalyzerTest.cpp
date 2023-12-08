//===--  Automemcpy Json Results Analyzer Test ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "automemcpy/ResultAnalyzer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::DoubleNear;
using testing::ElementsAre;
using testing::Pair;
using testing::SizeIs;

namespace llvm {
namespace automemcpy {
namespace {

TEST(AutomemcpyJsonResultsAnalyzer, getThroughputsOneSample) {
  static constexpr FunctionId Foo1 = {"memcpy1", FunctionType::MEMCPY};
  static constexpr DistributionId DistA = {{"A"}};
  static constexpr SampleId Id = {Foo1, DistA};
  static constexpr Sample kSamples[] = {
      Sample{Id, SampleType::ITERATION, 4},
      Sample{Id, SampleType::AGGREGATE, -1}, // Aggegates gets discarded
  };

  const std::vector<FunctionData> Data = getThroughputs(kSamples);
  EXPECT_THAT(Data, SizeIs(1));
  EXPECT_THAT(Data[0].Id, Foo1);
  EXPECT_THAT(Data[0].PerDistributionData, SizeIs(1));
  // A single value is provided.
  const auto &DistributionData = Data[0].PerDistributionData.lookup(DistA.Name);
  EXPECT_THAT(DistributionData.BytesPerSecondMedian, 4);
  EXPECT_THAT(DistributionData.BytesPerSecondMean, 4);
  EXPECT_THAT(DistributionData.BytesPerSecondVariance, 0);
}

TEST(AutomemcpyJsonResultsAnalyzer, getThroughputsManySamplesSameBucket) {
  static constexpr FunctionId Foo1 = {"memcpy1", FunctionType::MEMCPY};
  static constexpr DistributionId DistA = {{"A"}};
  static constexpr SampleId Id = {Foo1, DistA};
  static constexpr Sample kSamples[] = {Sample{Id, SampleType::ITERATION, 4},
                                        Sample{Id, SampleType::ITERATION, 5},
                                        Sample{Id, SampleType::ITERATION, 5}};

  const std::vector<FunctionData> Data = getThroughputs(kSamples);
  EXPECT_THAT(Data, SizeIs(1));
  EXPECT_THAT(Data[0].Id, Foo1);
  EXPECT_THAT(Data[0].PerDistributionData, SizeIs(1));
  // When multiple values are provided we pick the median one (here median of 4,
  // 5, 5).
  const auto &DistributionData = Data[0].PerDistributionData.lookup(DistA.Name);
  EXPECT_THAT(DistributionData.BytesPerSecondMedian, 5);
  EXPECT_THAT(DistributionData.BytesPerSecondMean, DoubleNear(4.6, 0.1));
  EXPECT_THAT(DistributionData.BytesPerSecondVariance, DoubleNear(0.33, 0.01));
}

TEST(AutomemcpyJsonResultsAnalyzer, getThroughputsServeralFunctionAndDist) {
  static constexpr FunctionId Foo1 = {"memcpy1", FunctionType::MEMCPY};
  static constexpr DistributionId DistA = {{"A"}};
  static constexpr FunctionId Foo2 = {"memcpy2", FunctionType::MEMCPY};
  static constexpr DistributionId DistB = {{"B"}};
  static constexpr Sample kSamples[] = {
      Sample{{Foo1, DistA}, SampleType::ITERATION, 1},
      Sample{{Foo1, DistB}, SampleType::ITERATION, 2},
      Sample{{Foo2, DistA}, SampleType::ITERATION, 3},
      Sample{{Foo2, DistB}, SampleType::ITERATION, 4}};
  // Data is aggregated per function.
  const std::vector<FunctionData> Data = getThroughputs(kSamples);
  EXPECT_THAT(Data, SizeIs(2)); // 2 functions Foo1 and Foo2.
  // Each function has data for both distributions DistA and DistB.
  EXPECT_THAT(Data[0].PerDistributionData, SizeIs(2));
  EXPECT_THAT(Data[1].PerDistributionData, SizeIs(2));
}

TEST(AutomemcpyJsonResultsAnalyzer, getScore) {
  static constexpr FunctionId Foo1 = {"memcpy1", FunctionType::MEMCPY};
  static constexpr FunctionId Foo2 = {"memcpy2", FunctionType::MEMCPY};
  static constexpr FunctionId Foo3 = {"memcpy3", FunctionType::MEMCPY};
  static constexpr DistributionId Dist = {{"A"}};
  static constexpr Sample kSamples[] = {
      Sample{{Foo1, Dist}, SampleType::ITERATION, 1},
      Sample{{Foo2, Dist}, SampleType::ITERATION, 2},
      Sample{{Foo3, Dist}, SampleType::ITERATION, 3}};

  // Data is aggregated per function.
  std::vector<FunctionData> Data = getThroughputs(kSamples);

  // Sort Data by function name so we can test them.
  std::sort(
      Data.begin(), Data.end(),
      [](const FunctionData &A, const FunctionData &B) { return A.Id < B.Id; });

  EXPECT_THAT(Data[0].Id, Foo1);
  EXPECT_THAT(Data[0].PerDistributionData.lookup("A").BytesPerSecondMedian, 1);
  EXPECT_THAT(Data[1].Id, Foo2);
  EXPECT_THAT(Data[1].PerDistributionData.lookup("A").BytesPerSecondMedian, 2);
  EXPECT_THAT(Data[2].Id, Foo3);
  EXPECT_THAT(Data[2].PerDistributionData.lookup("A").BytesPerSecondMedian, 3);

  // Normalizes throughput per distribution.
  fillScores(Data);
  EXPECT_THAT(Data[0].PerDistributionData.lookup("A").Score, 0);
  EXPECT_THAT(Data[1].PerDistributionData.lookup("A").Score, 0.5);
  EXPECT_THAT(Data[2].PerDistributionData.lookup("A").Score, 1);
}

TEST(AutomemcpyJsonResultsAnalyzer, castVotes) {
  static constexpr double kAbsErr = 0.01;

  static constexpr FunctionId Foo1 = {"memcpy1", FunctionType::MEMCPY};
  static constexpr FunctionId Foo2 = {"memcpy2", FunctionType::MEMCPY};
  static constexpr FunctionId Foo3 = {"memcpy3", FunctionType::MEMCPY};
  static constexpr DistributionId DistA = {{"A"}};
  static constexpr DistributionId DistB = {{"B"}};
  static constexpr Sample kSamples[] = {
      Sample{{Foo1, DistA}, SampleType::ITERATION, 0},
      Sample{{Foo1, DistB}, SampleType::ITERATION, 30},
      Sample{{Foo2, DistA}, SampleType::ITERATION, 1},
      Sample{{Foo2, DistB}, SampleType::ITERATION, 100},
      Sample{{Foo3, DistA}, SampleType::ITERATION, 7},
      Sample{{Foo3, DistB}, SampleType::ITERATION, 100},
  };

  // DistA Thoughput ranges from 0 to 7.
  // DistB Thoughput ranges from 30 to 100.

  // Data is aggregated per function.
  std::vector<FunctionData> Data = getThroughputs(kSamples);

  // Sort Data by function name so we can test them.
  std::sort(
      Data.begin(), Data.end(),
      [](const FunctionData &A, const FunctionData &B) { return A.Id < B.Id; });

  // Normalizes throughput per distribution.
  fillScores(Data);

  // Cast votes
  castVotes(Data);

  EXPECT_THAT(Data[0].Id, Foo1);
  EXPECT_THAT(Data[1].Id, Foo2);
  EXPECT_THAT(Data[2].Id, Foo3);

  const auto GetDistData = [&Data](size_t Index, StringRef Name) {
    return Data[Index].PerDistributionData.lookup(Name);
  };

  // Distribution A
  // Throughput is 0, 1 and 7, so normalized scores are 0, 1/7 and 1.
  EXPECT_THAT(GetDistData(0, "A").Score, DoubleNear(0, kAbsErr));
  EXPECT_THAT(GetDistData(1, "A").Score, DoubleNear(1. / 7, kAbsErr));
  EXPECT_THAT(GetDistData(2, "A").Score, DoubleNear(1, kAbsErr));
  // which are turned into grades BAD,  MEDIOCRE and EXCELLENT.
  EXPECT_THAT(GetDistData(0, "A").Grade, Grade::BAD);
  EXPECT_THAT(GetDistData(1, "A").Grade, Grade::MEDIOCRE);
  EXPECT_THAT(GetDistData(2, "A").Grade, Grade::EXCELLENT);

  // Distribution B
  // Throughput is 30, 100 and 100, so normalized scores are 0, 1 and 1.
  EXPECT_THAT(GetDistData(0, "B").Score, DoubleNear(0, kAbsErr));
  EXPECT_THAT(GetDistData(1, "B").Score, DoubleNear(1, kAbsErr));
  EXPECT_THAT(GetDistData(2, "B").Score, DoubleNear(1, kAbsErr));
  // which are turned into grades BAD, EXCELLENT and EXCELLENT.
  EXPECT_THAT(GetDistData(0, "B").Grade, Grade::BAD);
  EXPECT_THAT(GetDistData(1, "B").Grade, Grade::EXCELLENT);
  EXPECT_THAT(GetDistData(2, "B").Grade, Grade::EXCELLENT);

  // Now looking from the functions point of view.
  EXPECT_THAT(Data[0].ScoresGeoMean, DoubleNear(0, kAbsErr));
  EXPECT_THAT(Data[1].ScoresGeoMean, DoubleNear(1. * (1. / 7), kAbsErr));
  EXPECT_THAT(Data[2].ScoresGeoMean, DoubleNear(1, kAbsErr));

  // Note the array is indexed by GradeEnum values (EXCELLENT=0 / BAD = 6)
  EXPECT_THAT(Data[0].GradeHisto, ElementsAre(0, 0, 0, 0, 0, 0, 2));
  EXPECT_THAT(Data[1].GradeHisto, ElementsAre(1, 0, 0, 0, 0, 1, 0));
  EXPECT_THAT(Data[2].GradeHisto, ElementsAre(2, 0, 0, 0, 0, 0, 0));

  EXPECT_THAT(Data[0].FinalGrade, Grade::BAD);
  EXPECT_THAT(Data[1].FinalGrade, Grade::MEDIOCRE);
  EXPECT_THAT(Data[2].FinalGrade, Grade::EXCELLENT);
}

} // namespace
} // namespace automemcpy
} // namespace llvm
