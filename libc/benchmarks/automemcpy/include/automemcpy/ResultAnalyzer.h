//===-- Analyze benchmark JSON files ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_BENCHMARKS_AUTOMEMCPY_RESULTANALYZER_H
#define LIBC_BENCHMARKS_AUTOMEMCPY_RESULTANALYZER_H

#include "automemcpy/FunctionDescriptor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include <array>
#include <vector>

namespace llvm {
namespace automemcpy {

// A Grade as in the Majority Judgment voting system.
struct Grade {
  enum GradeEnum {
    EXCELLENT,
    VERY_GOOD,
    GOOD,
    PASSABLE,
    INADEQUATE,
    MEDIOCRE,
    BAD,
    ARRAY_SIZE,
  };

  // Returns a human readable string of the enum.
  static StringRef getString(const GradeEnum &GE);

  // Turns 'Score' into a GradeEnum.
  static GradeEnum judge(double Score);
};

// A 'GradeEnum' indexed array with counts for each grade.
using GradeHistogram = std::array<size_t, Grade::ARRAY_SIZE>;

// Identifies a Function by its name and type. Used as a key in a map.
struct FunctionId {
  StringRef Name;
  FunctionType Type;
  COMPARABLE_AND_HASHABLE(FunctionId, Type, Name)
};

struct PerDistributionData {
  std::vector<double> BytesPerSecondSamples;
  double BytesPerSecondMedian;   // Median of samples for this distribution.
  double BytesPerSecondMean;     // Mean of samples for this distribution.
  double BytesPerSecondVariance; // Variance of samples for this distribution.
  double Score;                  // Normalized score for this distribution.
  Grade::GradeEnum Grade;        // Grade for this distribution.
};

struct FunctionData {
  FunctionId Id;
  StringMap<PerDistributionData> PerDistributionData;
  double ScoresGeoMean;           // Geomean of scores for each distribution.
  GradeHistogram GradeHisto = {}; // GradeEnum indexed array
  Grade::GradeEnum FinalGrade = Grade::BAD; // Overall grade for this function
};

// Identifies a Distribution by its name. Used as a key in a map.
struct DistributionId {
  StringRef Name;
  COMPARABLE_AND_HASHABLE(DistributionId, Name)
};

// Identifies a Sample by its distribution and function. Used as a key in a map.
struct SampleId {
  FunctionId Function;
  DistributionId Distribution;
  COMPARABLE_AND_HASHABLE(SampleId, Function.Type, Function.Name,
                          Distribution.Name)
};

// The type of Samples as reported by the Google Benchmark's JSON result file.
// We are only interested in the "iteration" samples, the "aggregate" ones
// represent derived metrics such as 'mean' or 'median'.
enum class SampleType { UNKNOWN, ITERATION, AGGREGATE };

// A SampleId with an associated measured throughput.
struct Sample {
  SampleId Id;
  SampleType Type = SampleType::UNKNOWN;
  double BytesPerSecond = 0;
};

// This function collects Samples that belong to the same distribution and
// function and retains the median one. It then stores each of them into a
// 'FunctionData' and returns them as a vector.
std::vector<FunctionData> getThroughputs(ArrayRef<Sample> Samples);

// Normalize the function's throughput per distribution.
void fillScores(MutableArrayRef<FunctionData> Functions);

// Convert scores into Grades, stores an histogram of Grade for each functions
// and cast a median grade for the function.
void castVotes(MutableArrayRef<FunctionData> Functions);

} // namespace automemcpy
} // namespace llvm

#endif // LIBC_BENCHMARKS_AUTOMEMCPY_RESULTANALYZER_H
