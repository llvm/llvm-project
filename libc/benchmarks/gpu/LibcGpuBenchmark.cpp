#include "LibcGpuBenchmark.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/string.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/fixedvector.h"
#include "src/time/gpu/time_utils.h"

namespace LIBC_NAMESPACE {
namespace benchmarks {

FixedVector<Benchmark *, 64> benchmarks;
cpp::array<BenchmarkResult, 1024> results;

void Benchmark::add_benchmark(Benchmark *benchmark) {
  benchmarks.push_back(benchmark);
}

BenchmarkResult reduce_results(cpp::array<BenchmarkResult, 1024> &results) {
  BenchmarkResult result;
  uint64_t cycles_sum = 0;
  double standard_deviation_sum = 0;
  uint64_t min = UINT64_MAX;
  uint64_t max = 0;
  uint32_t samples_sum = 0;
  uint32_t iterations_sum = 0;
  clock_t time_sum = 0;
  uint64_t num_threads = gpu::get_num_threads();
  for (uint64_t i = 0; i < num_threads; i++) {
    BenchmarkResult current_result = results[i];
    cycles_sum += current_result.cycles;
    standard_deviation_sum += current_result.standard_deviation;
    min = cpp::min(min, current_result.min);
    max = cpp::max(max, current_result.max);
    samples_sum += current_result.samples;
    iterations_sum += current_result.total_iterations;
    time_sum += current_result.total_time;
  }
  result.cycles = cycles_sum / num_threads;
  result.standard_deviation = standard_deviation_sum / num_threads;
  result.min = min;
  result.max = max;
  result.samples = samples_sum / num_threads;
  result.total_iterations = iterations_sum / num_threads;
  result.total_time = time_sum / num_threads;
  return result;
}

void Benchmark::run_benchmarks() {
  uint64_t id = gpu::get_thread_id();
  gpu::sync_threads();

  for (Benchmark *benchmark : benchmarks)
    results[id] = benchmark->run();
  gpu::sync_threads();
  if (id == 0) {
    for (Benchmark *benchmark : benchmarks) {
      BenchmarkResult all_results = reduce_results(results);
      constexpr auto GREEN = "\033[32m";
      constexpr auto RESET = "\033[0m";
      log << GREEN << "[ RUN      ] " << RESET << benchmark->get_name() << '\n';
      log << GREEN << "[       OK ] " << RESET << benchmark->get_name() << ": "
          << all_results.cycles << " cycles, " << all_results.min << " min, "
          << all_results.max << " max, " << all_results.total_iterations
          << " iterations, " << all_results.total_time << " ns, "
          << static_cast<long>(all_results.standard_deviation) << " stddev\n";
    }
  }
  gpu::sync_threads();
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
  result.standard_deviation = fputil::sqrt<double>(
      static_cast<double>(cycles_squared) / total_iterations -
      static_cast<double>(best_guess * best_guess));
  result.min = min;
  result.max = max;
  result.samples = samples;
  result.total_iterations = total_iterations;
  result.total_time = total_time;
  return result;
};

} // namespace benchmarks
} // namespace LIBC_NAMESPACE
