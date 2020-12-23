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
      "StudyName",
      Runtime{HostState{"CpuName",
                        123,
                        {CacheInfo{"A", 1, 2, 3}, CacheInfo{"B", 4, 5, 6}}},
              456, 789,
              BenchmarkOptions{std::chrono::seconds(1), std::chrono::seconds(2),
                               10, 100, 6, 100, 0.1, 2, BenchmarkLog::Full}},
      StudyConfiguration{std::string("Function"), 30U, false, 32U,
                         std::string("Distribution"), Align(16), 3U},
      {std::chrono::seconds(3), std::chrono::seconds(4)}};
}

static std::string serializeToString(const Study &S) {
  std::string Buffer;
  raw_string_ostream RSO(Buffer);
  json::OStream JOS(RSO);
  serializeToJson(S, JOS);
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

auto equals(const HostState &H) -> auto {
  return AllOf(
      Field(&HostState::CpuName, H.CpuName),
      Field(&HostState::CpuFrequency, H.CpuFrequency),
      Field(&HostState::Caches, Pointwise(EqualsCacheInfo(), H.Caches)));
}

auto equals(const StudyConfiguration &SC) -> auto {
  return AllOf(
      Field(&StudyConfiguration::Function, SC.Function),
      Field(&StudyConfiguration::NumTrials, SC.NumTrials),
      Field(&StudyConfiguration::IsSweepMode, SC.IsSweepMode),
      Field(&StudyConfiguration::SweepModeMaxSize, SC.SweepModeMaxSize),
      Field(&StudyConfiguration::SizeDistributionName, SC.SizeDistributionName),
      Field(&StudyConfiguration::AccessAlignment, SC.AccessAlignment),
      Field(&StudyConfiguration::MemcmpMismatchAt, SC.MemcmpMismatchAt));
}

auto equals(const BenchmarkOptions &BO) -> auto {
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

auto equals(const Runtime &RI) -> auto {
  return AllOf(Field(&Runtime::Host, equals(RI.Host)),
               Field(&Runtime::BufferSize, RI.BufferSize),
               Field(&Runtime::BatchParameterCount, RI.BatchParameterCount),
               Field(&Runtime::BenchmarkOptions, equals(RI.BenchmarkOptions)));
}

auto equals(const Study &S) -> auto {
  return AllOf(Field(&Study::StudyName, S.StudyName),
               Field(&Study::Runtime, equals(S.Runtime)),
               Field(&Study::Configuration, equals(S.Configuration)),
               Field(&Study::Measurements, S.Measurements));
}

TEST(JsonTest, RoundTrip) {
  const Study S = getStudy();
  const auto Serialized = serializeToString(S);
  auto StudyOrError = parseJsonStudy(Serialized);
  if (auto Err = StudyOrError.takeError())
    EXPECT_FALSE(Err) << "Unexpected error : " << Err << "\n" << Serialized;
  const Study &Parsed = *StudyOrError;
  EXPECT_THAT(Parsed, equals(S)) << Serialized << "\n"
                                 << serializeToString(Parsed);
}

TEST(JsonTest, SupplementaryField) {
  auto Failure = parseJsonStudy(R"({
      "UnknownField": 10
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()), "Unknown field: UnknownField");
}

TEST(JsonTest, InvalidType) {
  auto Failure = parseJsonStudy(R"({
      "Runtime": 1
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()), "Expected JSON Object");
}

TEST(JsonTest, InvalidDuration) {
  auto Failure = parseJsonStudy(R"({
      "Runtime": {
        "BenchmarkOptions": {
          "MinDuration": "Duration should be a Number"
        }
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()), "Can't parse Duration");
}

TEST(JsonTest, InvalidAlignType) {
  auto Failure = parseJsonStudy(R"({
      "Configuration": {
        "AccessAlignment": "Align should be an Integer"
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()), "Can't parse Align, not an Integer");
}

TEST(JsonTest, InvalidAlign) {
  auto Failure = parseJsonStudy(R"({
      "Configuration": {
        "AccessAlignment": 3
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()),
            "Can't parse Align, not a power of two");
}

TEST(JsonTest, InvalidBenchmarkLogType) {
  auto Failure = parseJsonStudy(R"({
      "Runtime": {
        "BenchmarkOptions":{
          "Log": 3
        }
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()),
            "Can't parse BenchmarkLog, not a String");
}

TEST(JsonTest, InvalidBenchmarkLog) {
  auto Failure = parseJsonStudy(R"({
      "Runtime": {
        "BenchmarkOptions":{
          "Log": "Unknown"
        }
      }
    }
  )");
  EXPECT_EQ(toString(Failure.takeError()),
            "Can't parse BenchmarkLog, invalid value 'Unknown'");
}

} // namespace
} // namespace libc_benchmarks
} // namespace llvm
