//===-- Benchmark function tests -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcBenchmark.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>
#include <limits>
#include <optional>
#include <queue>
#include <vector>

using std::chrono::nanoseconds;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::SizeIs;

namespace llvm {
namespace libc_benchmarks {
namespace {

// A simple parameter provider returning a zero initialized vector of size
// `iterations`.
struct DummyParameterProvider {
  std::vector<char> generate_batch(size_t iterations) {
    return std::vector<char>(iterations);
  }
};

class LibcBenchmark : public ::testing::Test {
public:
  // A Clock interface suitable for testing.
  // - Either it returns 0,
  // - Or a timepoint coming from the `setMeasurements` call.
  Duration now() {
    if (!MaybeTimepoints)
      return {};
    assert(!MaybeTimepoints->empty());
    const Duration timepoint = MaybeTimepoints->front();
    MaybeTimepoints->pop();
    return timepoint;
  }

protected:
  void SetUp() override { options.log = BenchmarkLog::Full; }

  void TearDown() override {
    // We make sure all the expected measurements were performed.
    if (MaybeTimepoints) {
      EXPECT_THAT(*MaybeTimepoints, IsEmpty());
    }
  }

  BenchmarkResult run() {
    return benchmark(options, ParameterProvider, DummyFunction, *this);
  }

  void setMeasurements(llvm::ArrayRef<Duration> Durations) {
    MaybeTimepoints.emplace(); // Create the optional value.
    Duration CurrentTime = nanoseconds(1);
    for (const auto &Duration : Durations) {
      MaybeTimepoints->push(CurrentTime);
      CurrentTime += Duration;
      MaybeTimepoints->push(CurrentTime);
      CurrentTime += nanoseconds(1);
    }
  }

  BenchmarkOptions options;

private:
  DummyParameterProvider ParameterProvider;
  static char DummyFunction(char Payload) { return Payload; }
  std::optional<std::queue<Duration>> MaybeTimepoints;
};

TEST_F(LibcBenchmark, MaxSamplesReached) {
  options.max_samples = 1;
  const auto result = run();
  EXPECT_THAT(result.maybe_benchmark_log->size(), 1);
  EXPECT_THAT(result.termination_status, BenchmarkStatus::MaxSamplesReached);
}

TEST_F(LibcBenchmark, MaxDurationReached) {
  options.max_duration = nanoseconds(10);
  setMeasurements({nanoseconds(11)});
  const auto result = run();
  EXPECT_THAT(result.maybe_benchmark_log->size(), 1);
  EXPECT_THAT(result.termination_status, BenchmarkStatus::MaxDurationReached);
}

TEST_F(LibcBenchmark, MaxIterationsReached) {
  options.initial_iterations = 1;
  options.max_iterations = 20;
  options.scaling_factor = 2;
  options.epsilon = 0; // unreachable.
  const auto result = run();
  EXPECT_THAT(*result.maybe_benchmark_log,
              ElementsAre(Field(&BenchmarkState::last_sample_iterations, 1),
                          Field(&BenchmarkState::last_sample_iterations, 2),
                          Field(&BenchmarkState::last_sample_iterations, 4),
                          Field(&BenchmarkState::last_sample_iterations, 8),
                          Field(&BenchmarkState::last_sample_iterations, 16),
                          Field(&BenchmarkState::last_sample_iterations, 32)));
  EXPECT_THAT(result.maybe_benchmark_log->size(), 6);
  EXPECT_THAT(result.termination_status, BenchmarkStatus::MaxIterationsReached);
}

TEST_F(LibcBenchmark, MinSamples) {
  options.min_samples = 4;
  options.scaling_factor = 2;
  options.epsilon = std::numeric_limits<double>::max(); // always reachable.
  setMeasurements(
      {nanoseconds(1), nanoseconds(2), nanoseconds(4), nanoseconds(8)});
  const auto result = run();
  EXPECT_THAT(*result.maybe_benchmark_log,
              ElementsAre(Field(&BenchmarkState::last_sample_iterations, 1),
                          Field(&BenchmarkState::last_sample_iterations, 2),
                          Field(&BenchmarkState::last_sample_iterations, 4),
                          Field(&BenchmarkState::last_sample_iterations, 8)));
  EXPECT_THAT(result.maybe_benchmark_log->size(), 4);
  EXPECT_THAT(result.termination_status, BenchmarkStatus::PrecisionReached);
}

TEST_F(LibcBenchmark, Epsilon) {
  options.min_samples = 4;
  options.scaling_factor = 2;
  options.epsilon = std::numeric_limits<double>::max(); // always reachable.
  setMeasurements(
      {nanoseconds(1), nanoseconds(2), nanoseconds(4), nanoseconds(8)});
  const auto result = run();
  EXPECT_THAT(*result.maybe_benchmark_log,
              ElementsAre(Field(&BenchmarkState::last_sample_iterations, 1),
                          Field(&BenchmarkState::last_sample_iterations, 2),
                          Field(&BenchmarkState::last_sample_iterations, 4),
                          Field(&BenchmarkState::last_sample_iterations, 8)));
  EXPECT_THAT(result.maybe_benchmark_log->size(), 4);
  EXPECT_THAT(result.termination_status, BenchmarkStatus::PrecisionReached);
}

TEST(ArrayRefLoop, Cycle) {
  std::array<int, 2> array = {1, 2};
  EXPECT_THAT(cycle(array, 0), ElementsAre());
  EXPECT_THAT(cycle(array, 1), ElementsAre(1));
  EXPECT_THAT(cycle(array, 2), ElementsAre(1, 2));
  EXPECT_THAT(cycle(array, 3), ElementsAre(1, 2, 1));
  EXPECT_THAT(cycle(array, 4), ElementsAre(1, 2, 1, 2));
  EXPECT_THAT(cycle(array, 5), ElementsAre(1, 2, 1, 2, 1));
}

TEST(ByteConstrainedArray, Simple) {
  EXPECT_THAT((ByteConstrainedArray<char, 17>()), SizeIs(17));
  EXPECT_THAT((ByteConstrainedArray<uint16_t, 17>()), SizeIs(8));
  EXPECT_THAT((ByteConstrainedArray<uint32_t, 17>()), SizeIs(4));
  EXPECT_THAT((ByteConstrainedArray<uint64_t, 17>()), SizeIs(2));

  EXPECT_LE(sizeof(ByteConstrainedArray<char, 17>), 17U);
  EXPECT_LE(sizeof(ByteConstrainedArray<uint16_t, 17>), 17U);
  EXPECT_LE(sizeof(ByteConstrainedArray<uint32_t, 17>), 17U);
  EXPECT_LE(sizeof(ByteConstrainedArray<uint64_t, 17>), 17U);
}

TEST(ByteConstrainedArray, Cycle) {
  ByteConstrainedArray<uint64_t, 17> TwoValues{{1UL, 2UL}};
  EXPECT_THAT(cycle(TwoValues, 5), ElementsAre(1, 2, 1, 2, 1));
}
} // namespace
} // namespace libc_benchmarks
} // namespace llvm
