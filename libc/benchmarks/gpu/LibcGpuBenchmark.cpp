#include "LibcGpuBenchmark.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/time/gpu/time_utils.h"

namespace LIBC_NAMESPACE {
namespace benchmarks {

FixedVector<Benchmark *, 64> benchmarks;

void Benchmark::add_benchmark(Benchmark *benchmark) {
  benchmarks.push_back(benchmark);
}

void Benchmark::run_benchmarks() {
  for (Benchmark *benchmark : benchmarks)
    benchmark->run();
}

BenchmarkResult benchmark(const BenchmarkOptions &options,
                          cpp::function<uint64_t(void)> wrapper_func) {
  BenchmarkResult result;
  RuntimeEstimationProgression rep;
  uint32_t total_iterations = 0;
  uint32_t iterations = options.initial_iterations;
  if (iterations < 1u)
    iterations = 1;

  uint32_t samples = 0;
  uint64_t total_time = 0;
  uint64_t best_guess = 0;
  uint64_t total_cycles = 0;
  uint64_t cycles_squared = 0;
  uint64_t min = UINT64_MAX;
  uint64_t max = 0;

  uint64_t overhead = UINT64_MAX;
  int overhead_iterations = 10;
  for (int i = 0; i < overhead_iterations; i++)
    overhead = cpp::min(overhead, LIBC_NAMESPACE::overhead());

  for (uint64_t time_budget = options.max_duration; time_budget >= 0;) {
    uint64_t sample_cycles = 0;
    const clock_t start = static_cast<double>(clock());
    for (uint32_t i = 0; i < iterations; i++) {
      auto wrapper_intermediate = wrapper_func();
      uint64_t result = wrapper_intermediate - overhead;
      max = cpp::max(max, result);
      min = cpp::min(min, result);
      sample_cycles += result;
    }
    const clock_t end = clock();
    const clock_t duration_ns =
        ((end - start) * 1000 * 1000 * 1000) / CLOCKS_PER_SEC;
    total_time += duration_ns;
    time_budget -= duration_ns;
    samples++;
    total_cycles += sample_cycles;
    cycles_squared += sample_cycles * sample_cycles;

    total_iterations += iterations;
    const double change_ratio =
        rep.compute_improvement({iterations, sample_cycles});
    best_guess = rep.current_estimation;

    if (samples >= options.max_samples || iterations >= options.max_iterations)
      break;
    if (total_time >= options.min_duration && samples >= options.min_samples &&
        change_ratio < options.epsilon)
      break;

    iterations *= options.scaling_factor;
  }
  result.cycles = best_guess;
  result.standard_deviation =
      fputil::sqrt(static_cast<double>(cycles_squared) / total_iterations -
                   (best_guess * best_guess));
  result.min = min;
  result.max = max;
  result.samples = samples;
  result.total_iterations = total_iterations;
  result.total_time = total_time;
  return result;
};

} // namespace benchmarks
} // namespace LIBC_NAMESPACE
