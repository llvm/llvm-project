//===-- JSON Tests --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSON.h"
#include "LibcBenchmark.h"
#include "LibcMemoryBenchmark.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::AllOf;
using testing::ExplainMatchResult;
using testing::Field;
using testing::Pointwise;

namespace llvm {
namespace libc_benchmarks {
namespace {

Study getStudy() {
  return Study{
      HostState{
          "CpuName", 123, {CacheInfo{"A", 1, 2, 3}, CacheInfo{"B", 4, 5, 6}}},
      BenchmarkOptions{std::chrono::seconds(1), std::chrono::seconds(2), 10,
                       100, 6, 100, 0.1, 2, BenchmarkLog::Full},
      StudyConfiguration{2, 3, SizeRange{4, 5, 6}, Align(8), 9, 10},
      {FunctionMeasurements{"A",
                            {Measurement{3, std::chrono::seconds(3)},
                             Measurement{3, std::chrono::seconds(4)}}},
       FunctionMeasurements{"B", {}}}};
}

static std::string SerializeToString(const Study &S) {
  std::string Buffer;
  raw_string_ostream RSO(Buffer);
  json::OStream JOS(RSO);
  SerializeToJson(S, JOS);
  return Buffer;
}

MATCHER(EqualsCacheInfo, "") {
  const CacheInfo &A = ::testing::get<0>(arg);
  const CacheInfo &B = ::testing::get<1>(arg);
  return ExplainMatchResult(AllOf(Field(&CacheInfo::Type, B.Type),
                                  Field(&CacheInfo::Level, B.Level),
                                  Field(&CacheInfo::Size, B.Size),
                                  Field(&CacheInfo::NumSharing, B.NumSharing)),
                            A, result_listener);
}

auto Equals(const HostState &H) -> auto {
  return AllOf(
      Field(&HostState::CpuName, H.CpuName),
      Field(&HostState::CpuFrequency, H.CpuFrequency),
      Field(&HostState::Caches, Pointwise(EqualsCacheInfo(), H.Caches)));
}

auto Equals(const BenchmarkOptions &BO) -> auto {
  return AllOf(
      Field(&BenchmarkOptions::MinDuration, BO.MinDuration),
      Field(&BenchmarkOptions::MaxDuration, BO.MaxDuration),
      Field(&BenchmarkOptions::InitialIterations, BO.InitialIterations),
      Field(&BenchmarkOptions::MaxIterations, BO.MaxIterations),
      Field(&BenchmarkOptions::MinSamples, BO.MinSamples),
      Field(&BenchmarkOptions::MaxSamples, BO.MaxSamples),
      Field(&BenchmarkOptions::Epsilon, BO.Epsilon),
      Field(&BenchmarkOptions::ScalingFactor, BO.ScalingFactor),
      Field(&BenchmarkOptions::Log, BO.Log));
}

auto Equals(const SizeRange &SR) -> auto {
  return AllOf(Field(&SizeRange::From, SR.From), Field(&SizeRange::To, SR.To),
               Field(&SizeRange::Step, SR.Step));
}

auto Equals(const StudyConfiguration &SC) -> auto {
  return AllOf(
      Field(&StudyConfiguration::Runs, SC.Runs),
      Field(&StudyConfiguration::BufferSize, SC.BufferSize),
      Field(&StudyConfiguration::Size, Equals(SC.Size)),
      Field(&StudyConfiguration::AddressAlignment, SC.AddressAlignment),
      Field(&StudyConfiguration::MemsetValue, SC.MemsetValue),
      Field(&StudyConfiguration::MemcmpMismatchAt, SC.MemcmpMismatchAt));
}

MATCHER(EqualsMeasurement, "") {
  const Measurement &A = ::testing::get<0>(arg);
  const Measurement &B = ::testing::get<1>(arg);
  return ExplainMatchResult(AllOf(Field(&Measurement::Size, B.Size),
                                  Field(&Measurement::Runtime, B.Runtime)),
                            A, result_listener);
}

MATCHER(EqualsFunctions, "") {
  const FunctionMeasurements &A = ::testing::get<0>(arg);
  const FunctionMeasurements &B = ::testing::get<1>(arg);
  return ExplainMatchResult(
      AllOf(Field(&FunctionMeasurements::Name, B.Name),
            Field(&FunctionMeasurements::Measurements,
                  Pointwise(EqualsMeasurement(), B.Measurements))),
      A, result_listener);
}

auto Equals(const Study &S) -> auto {
  return AllOf(
      Field(&Study::Host, Equals(S.Host)),
      Field(&Study::Options, Equals(S.Options)),
      Field(&Study::Configuration, Equals(S.Configuration)),
      Field(&Study::Functions, Pointwise(EqualsFunctions(), S.Functions)));
}

TEST(JsonTest, RoundTrip) {
  const Study S = getStudy();
  auto StudyOrError = ParseJsonStudy(SerializeToString(S));
  if (auto Err = StudyOrError.takeError())
    EXPECT_FALSE(Err) << "Unexpected error";
  const Study &Parsed = *StudyOrError;
  EXPECT_THAT(Parsed, Equals(S));
}

TEST(JsonTest, SupplementaryField) {
  auto Failure = ParseJsonStudy(R"({
      "UnknownField": 10
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()), "Unknown field: UnknownField");
}

TEST(JsonTest, InvalidType) {
  auto Failure = ParseJsonStudy(R"({
      "Options": 1
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()), "Expected JSON Object");
}

TEST(JsonTest, InvalidDuration) {
  auto Failure = ParseJsonStudy(R"({
      "Options": {
        "MinDuration": "Duration should be a Number"
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()), "Can't parse Duration");
}

TEST(JsonTest, InvalidAlignType) {
  auto Failure = ParseJsonStudy(R"({
      "Configuration":{
        "AddressAlignment": "Align should be an Integer"
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()), "Can't parse Align, not an Integer");
}

TEST(JsonTest, InvalidAlign) {
  auto Failure = ParseJsonStudy(R"({
      "Configuration":{
        "AddressAlignment":3
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()),
            "Can't parse Align, not a power of two");
}

TEST(JsonTest, InvalidBenchmarkLogType) {
  auto Failure = ParseJsonStudy(R"({
      "Options":{
        "Log": 3
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()),
            "Can't parse BenchmarkLog, not a String");
}

TEST(JsonTest, InvalidBenchmarkLog) {
  auto Failure = ParseJsonStudy(R"({
      "Options":{
        "Log": "Unknown"
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()),
            "Can't parse BenchmarkLog, invalid value 'Unknown'");
}

} // namespace
} // namespace libc_benchmarks
} // namespace llvm
