//===-- Benchmark function --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file mainly defines a `Benchmark` function.
//
// The benchmarking process is as follows:
// - We start by measuring the time it takes to run the function
// `InitialIterations` times. This is called a Sample. From this we can derive
// the time it took to run a single iteration.
//
// - We repeat the previous step with a greater number of iterations to lower
// the impact of the measurement. We can derive a more precise estimation of the
// runtime for a single iteration.
//
// - Each sample gives a more accurate estimation of the runtime for a single
// iteration but also takes more time to run. We stop the process when:
//   * The measure stabilize under a certain precision (Epsilon),
//   * The overall benchmarking time is greater than MaxDuration,
//   * The overall sample count is greater than MaxSamples,
//   * The last sample used more than MaxIterations iterations.
//
// - We also makes sure that the benchmark doesn't run for a too short period of
// time by defining MinDuration and MinSamples.

#ifndef LLVM_LIBC_UTILS_BENCHMARK_BENCHMARK_H
#define LLVM_LIBC_UTILS_BENCHMARK_BENCHMARK_H

#include "benchmark/benchmark.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>

#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace libc_benchmarks {

using Duration = std::chrono::duration<double>;

enum class BenchmarkLog {
  None, // Don't keep the internal state of the benchmark.
  Last, // Keep only the last batch.
  Full  // Keep all iterations states, useful for testing or debugging.
};

// An object to configure the benchmark stopping conditions.
// See documentation at the beginning of the file for the overall algorithm and
// meaning of each field.
struct BenchmarkOptions {
  // The minimum time for which the benchmark is running.
  Duration min_duration = std::chrono::seconds(0);
  // The maximum time for which the benchmark is running.
  Duration max_duration = std::chrono::seconds(10);
  // The number of iterations in the first sample.
  uint32_t initial_iterations = 1;
  // The maximum number of iterations for any given sample.
  uint32_t max_iterations = 10000000;
  // The minimum number of samples.
  uint32_t min_samples = 4;
  // The maximum number of samples.
  uint32_t max_samples = 1000;
  // The benchmark will stop if the relative difference between the current and
  // the last estimation is less than epsilon. This is 1% by default.
  double epsilon = 0.01;
  // The number of iterations grows exponentially between each sample.
  // Must be greater or equal to 1.
  double scaling_factor = 1.4;
  BenchmarkLog log = BenchmarkLog::None;
};

// The state of a benchmark.
enum class BenchmarkStatus {
  Running,
  MaxDurationReached,
  MaxIterationsReached,
  MaxSamplesReached,
  PrecisionReached,
};

// The internal state of the benchmark, useful to debug, test or report
// statistics.
struct BenchmarkState {
  size_t last_sample_iterations;
  Duration last_batch_elapsed;
  BenchmarkStatus current_status;
  Duration current_best_guess; // The time estimation for a single run of `foo`.
  double change_ratio; // The change in time estimation between previous and
                       // current samples.
};

#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
using BenchmarkLogType = llvm::SmallVector<BenchmarkState, 16>;
#else
#include <vector>
using BenchmarkLogType = std::vector<BenchmarkState>;
#endif

// A lightweight result for a benchmark.
struct BenchmarkResult {
  BenchmarkStatus termination_status = BenchmarkStatus::Running;
  Duration best_guess = {};
  std::optional<BenchmarkLogType> maybe_benchmark_log;
};

// Stores information about a cache in the host memory system.
struct CacheInfo {
  std::string type; //  e.g. "Instruction", "Data", "Unified".
  int level;        // 0 is closest to processing unit.
  int size;         // In bytes.
  int num_sharing;  // The number of processing units (Hyper-Threading Thread)
                    // with which this cache is shared.
};

// Stores information about the host.
struct HostState {
  std::string cpu_name; // returns a string compatible with the -march option.
  double cpu_frequency; // in Hertz.
  std::vector<CacheInfo> caches;

  static HostState get();
};

namespace internal {

struct Measurement {
  size_t iterations = 0;
  Duration elapsed = {};
};

// Updates the estimation of the elapsed time for a single iteration.
class RefinableRuntimeEstimation {
  Duration total_time = {};
  size_t total_iterations = 0;

public:
  Duration update(const Measurement &m) {
    assert(m.iterations > 0);
    // Duration is encoded as a double (see definition).
    // `total_time` and `m.elapsed` are of the same magnitude so we don't expect
    // loss of precision due to radically different scales.
    total_time += m.elapsed;
    total_iterations += m.iterations;
    return total_time / total_iterations;
  }
};

// This class tracks the progression of the runtime estimation.
class RuntimeEstimationProgression {
  RefinableRuntimeEstimation rre;

public:
  Duration current_estimation = {};

  // Returns the change ratio between our best guess so far and the one from the
  // new measurement.
  double compute_improvement(const Measurement &m) {
    const Duration new_estimation = rre.update(m);
    const double ratio = fabs(((current_estimation / new_estimation) - 1.0));
    current_estimation = new_estimation;
    return ratio;
  }
};

} // namespace internal

// Measures the runtime of `foo` until conditions defined by `Options` are met.
//
// To avoid measurement's imprecisions we measure batches of `foo`.
// The batch size is growing by `ScalingFactor` to minimize the effect of
// measuring.
//
// Note: The benchmark is not responsible for serializing the executions of
// `foo`. It is not suitable for measuring, very small & side effect free
// functions, as the processor is free to execute several executions in
// parallel.
//
// - Options: A set of parameters controlling the stopping conditions for the
//     benchmark.
// - foo: The function under test. It takes one value and returns one value.
//     The input value is used to randomize the execution of `foo` as part of a
//     batch to mitigate the effect of the branch predictor. Signature:
//     `ProductType foo(ParameterProvider::value_type value);`
//     The output value is a product of the execution of `foo` and prevents the
//     compiler from optimizing out foo's body.
// - ParameterProvider: An object responsible for providing a range of
//     `Iterations` values to use as input for `foo`. The `value_type` of the
//     returned container has to be compatible with `foo` argument.
//     Must implement one of:
//     `Container<ParameterType> generateBatch(size_t Iterations);`
//     `const Container<ParameterType>& generateBatch(size_t Iterations);`
// - Clock: An object providing the current time. Must implement:
//     `std::chrono::time_point now();`
template <typename Function, typename ParameterProvider,
          typename BenchmarkClock = const std::chrono::high_resolution_clock>
BenchmarkResult benchmark(const BenchmarkOptions &options,
                          ParameterProvider &PP, Function foo,
                          BenchmarkClock &Clock = BenchmarkClock()) {
  BenchmarkResult result;
  internal::RuntimeEstimationProgression rep;
  Duration total_benchmark_duration = {};
  size_t iterations = std::max(options.initial_iterations, uint32_t(1));
  size_t samples = 0;
  if (options.scaling_factor < 1.0)
    report_fatal_error("scaling_factor should be >= 1");
  if (options.log != BenchmarkLog::None)
    result.maybe_benchmark_log.emplace();
  for (;;) {
    // Request a new Batch of size `iterations`.
    const auto &batch = PP.generate_batch(iterations);

    // Measuring this Batch.
    const auto start_time = Clock.now();
    for (const auto parameter : batch) {
      auto production = foo(parameter);
      benchmark::DoNotOptimize(production);
    }
    const auto end_time = Clock.now();
    const Duration elapsed = end_time - start_time;

    // Updating statistics.
    ++samples;
    total_benchmark_duration += elapsed;
    const double change_ratio = rep.compute_improvement({iterations, elapsed});
    result.best_guess = rep.current_estimation;

    // Stopping condition.
    if (total_benchmark_duration >= options.min_duration &&
        samples >= options.min_samples && change_ratio < options.epsilon)
      result.termination_status = BenchmarkStatus::PrecisionReached;
    else if (samples >= options.max_samples)
      result.termination_status = BenchmarkStatus::MaxSamplesReached;
    else if (total_benchmark_duration >= options.max_duration)
      result.termination_status = BenchmarkStatus::MaxDurationReached;
    else if (iterations >= options.max_iterations)
      result.termination_status = BenchmarkStatus::MaxIterationsReached;

    if (result.maybe_benchmark_log) {
      auto &benchmark_log = *result.maybe_benchmark_log;
      if (options.log == BenchmarkLog::Last && !benchmark_log.empty())
        benchmark_log.pop_back();
      BenchmarkState bs;
      bs.last_sample_iterations = iterations;
      bs.last_batch_elapsed = elapsed;
      bs.current_status = result.termination_status;
      bs.current_best_guess = result.best_guess;
      bs.change_ratio = change_ratio;
      benchmark_log.push_back(bs);
    }

    if (result.termination_status != BenchmarkStatus::Running)
      return result;

    if (options.scaling_factor > 1 &&
        iterations * options.scaling_factor == iterations)
      report_fatal_error("`iterations *= scaling_factor` is idempotent, "
                         "increase scaling_factor "
                         "or initial_iterations.");

    iterations *= options.scaling_factor;
  }
}

// Interprets `Array` as a circular buffer of `Size` elements.
template <typename T> class CircularArrayRef {
  llvm::ArrayRef<T> array;
  size_t size;

public:
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using difference_type = ssize_t;
  using size_type = size_t;

  class const_iterator {
    using iterator_category = std::input_iterator_tag;
    llvm::ArrayRef<T> array;
    size_t index;
    size_t offset;

  public:
    explicit const_iterator(llvm::ArrayRef<T> array, size_t index = 0)
        : array(array), index(index), offset(index % array.size()) {}
    const_iterator &operator++() {
      ++index;
      ++offset;
      if (offset == array.size())
        offset = 0;
      return *this;
    }
    bool operator==(const_iterator other) const { return index == other.index; }
    bool operator!=(const_iterator other) const { return !(*this == other); }
    const T &operator*() const { return array[offset]; }
  };

  CircularArrayRef(llvm::ArrayRef<T> array, size_t size)
      : array(array), size(size) {
    assert(array.size() > 0);
  }

  const_iterator begin() const { return const_iterator(array); }
  const_iterator end() const { return const_iterator(array, size); }
};

// A convenient helper to produce a CircularArrayRef from an ArrayRef.
template <typename T>
CircularArrayRef<T> cycle(llvm::ArrayRef<T> array, size_t size) {
  return {array, size};
}

// Creates an std::array which storage size is constrained under `Bytes`.
template <typename T, size_t Bytes>
using ByteConstrainedArray = std::array<T, Bytes / sizeof(T)>;

// A convenient helper to produce a CircularArrayRef from a
// ByteConstrainedArray.
template <typename T, size_t N>
CircularArrayRef<T> cycle(const std::array<T, N> &container, size_t size) {
  return {llvm::ArrayRef<T>(container.cbegin(), container.cend()), size};
}

// Makes sure the binary was compiled in release mode and that frequency
// governor is set on performance.
void checkRequirements();

} // namespace libc_benchmarks
} // namespace llvm

#endif // LLVM_LIBC_UTILS_BENCHMARK_BENCHMARK_H
