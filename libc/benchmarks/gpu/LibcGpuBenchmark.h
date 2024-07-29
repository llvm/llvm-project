#ifndef LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H
#define LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H

#include "benchmarks/gpu/BenchmarkLogger.h"
#include "benchmarks/gpu/timing/timing.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/time/clock.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

namespace benchmarks {

struct BenchmarkOptions {
  uint32_t initial_iterations = 1;
  uint32_t min_iterations = 50;
  uint32_t max_iterations = 10000000;
  uint32_t min_samples = 4;
  uint32_t max_samples = 1000;
  int64_t min_duration = 500 * 1000;         // 500 * 1000 nanoseconds = 500 us
  int64_t max_duration = 1000 * 1000 * 1000; // 1e9 nanoseconds = 1 second
  double epsilon = 0.0001;
  double scaling_factor = 1.4;
};

struct Measurement {
  uint32_t iterations = 0;
  uint64_t elapsed_cycles = 0;
};

class RefinableRuntimeEstimation {
  uint64_t total_cycles = 0;
  uint32_t total_iterations = 0;

public:
  uint64_t update(const Measurement &M) {
    total_cycles += M.elapsed_cycles;
    total_iterations += M.iterations;
    return total_cycles / total_iterations;
  }
};

// Tracks the progression of the runtime estimation
class RuntimeEstimationProgression {
  RefinableRuntimeEstimation rre;

public:
  uint64_t current_estimation = 0;

  double compute_improvement(const Measurement &M) {
    const uint64_t new_estimation = rre.update(M);
    double ratio =
        (static_cast<double>(current_estimation) / new_estimation) - 1.0;

    // Get absolute value
    if (ratio < 0)
      ratio *= -1;

    current_estimation = new_estimation;
    return ratio;
  }
};

struct BenchmarkResult {
  uint64_t cycles = 0;
  double standard_deviation = 0;
  uint64_t min = UINT64_MAX;
  uint64_t max = 0;
  uint32_t samples = 0;
  uint32_t total_iterations = 0;
  clock_t total_time = 0;
};

BenchmarkResult benchmark(const BenchmarkOptions &options,
                          cpp::function<uint64_t(void)> wrapper_func);

class Benchmark {
  const cpp::function<uint64_t(void)> func;
  const cpp::string_view suite_name;
  const cpp::string_view test_name;
  const uint32_t num_threads;

public:
  Benchmark(cpp::function<uint64_t(void)> func, char const *suite_name,
            char const *test_name, uint32_t num_threads)
      : func(func), suite_name(suite_name), test_name(test_name),
        num_threads(num_threads) {
    add_benchmark(this);
  }

  static void run_benchmarks();
  const cpp::string_view get_suite_name() const { return suite_name; }
  const cpp::string_view get_test_name() const { return test_name; }

protected:
  static void add_benchmark(Benchmark *benchmark);

private:
  BenchmarkResult run() {
    BenchmarkOptions options;
    return benchmark(options, func);
  }
};
} // namespace benchmarks
} // namespace LIBC_NAMESPACE_DECL

// Passing -1 indicates the benchmark should be run with as many threads as
// allocated by the user in the benchmark's CMake.
#define BENCHMARK(SuiteName, TestName, Func)                                   \
  LIBC_NAMESPACE::benchmarks::Benchmark SuiteName##_##TestName##_Instance(     \
      Func, #SuiteName, #TestName, -1)

#define BENCHMARK_N_THREADS(SuiteName, TestName, Func, NumThreads)             \
  LIBC_NAMESPACE::benchmarks::Benchmark SuiteName##_##TestName##_Instance(     \
      Func, #SuiteName, #TestName, NumThreads)

#define SINGLE_THREADED_BENCHMARK(SuiteName, TestName, Func)                   \
  BENCHMARK_N_THREADS(SuiteName, TestName, Func, 1)

#define SINGLE_WAVE_BENCHMARK(SuiteName, TestName, Func)                       \
  BENCHMARK_N_THREADS(SuiteName, TestName, Func,                               \
                      LIBC_NAMESPACE::gpu::get_lane_size())

#endif
