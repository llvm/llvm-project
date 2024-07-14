#include "LibcGpuBenchmark.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/string.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/fixedvector.h"
#include "src/__support/macros/config.h"
#include "src/time/gpu/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace benchmarks {

FixedVector<Benchmark *, 64> benchmarks;

void Benchmark::add_benchmark(Benchmark *benchmark) {
  benchmarks.push_back(benchmark);
}

void update_sums(const BenchmarkResult &current_result,
                 cpp::Atomic<uint64_t> &active_threads,
                 cpp::Atomic<uint64_t> &cycles_sum,
                 cpp::Atomic<uint64_t> &standard_deviation_sum,
                 cpp::Atomic<uint64_t> &min, cpp::Atomic<uint64_t> &max,
                 cpp::Atomic<uint32_t> &samples_sum,
                 cpp::Atomic<uint32_t> &iterations_sum,
                 cpp::Atomic<clock_t> &time_sum) {
  gpu::memory_fence();
  active_threads.fetch_add(1, cpp::MemoryOrder::RELAXED);

  cycles_sum.fetch_add(current_result.cycles, cpp::MemoryOrder::RELAXED);
  standard_deviation_sum.fetch_add(
      static_cast<uint64_t>(current_result.standard_deviation),
      cpp::MemoryOrder::RELAXED);

  // Perform a CAS loop to atomically update the min
  uint64_t orig_min = min.load(cpp::MemoryOrder::RELAXED);
  while (!min.compare_exchange_strong(
      orig_min, cpp::min(orig_min, current_result.min),
      cpp::MemoryOrder::ACQUIRE, cpp::MemoryOrder::RELAXED)) {
  }

  // Perform a CAS loop to atomically update the max
  uint64_t orig_max = max.load(cpp::MemoryOrder::RELAXED);
  while (!max.compare_exchange_strong(
      orig_max, cpp::max(orig_max, current_result.max),
      cpp::MemoryOrder::ACQUIRE, cpp::MemoryOrder::RELAXED)) {
  }

  samples_sum.fetch_add(current_result.samples, cpp::MemoryOrder::RELAXED);
  iterations_sum.fetch_add(current_result.total_iterations,
                           cpp::MemoryOrder::RELAXED);
  time_sum.fetch_add(current_result.total_time, cpp::MemoryOrder::RELAXED);
  gpu::memory_fence();
}

cpp::Atomic<uint64_t> cycles_sum = 0;
cpp::Atomic<uint64_t> standard_deviation_sum = 0;
cpp::Atomic<uint64_t> min = UINT64_MAX;
cpp::Atomic<uint64_t> max = 0;
cpp::Atomic<uint32_t> samples_sum = 0;
cpp::Atomic<uint32_t> iterations_sum = 0;
cpp::Atomic<clock_t> time_sum = 0;
cpp::Atomic<uint64_t> active_threads = 0;

void print_results(Benchmark *b) {
  constexpr auto GREEN = "\033[32m";
  constexpr auto RESET = "\033[0m";

  BenchmarkResult result;
  gpu::memory_fence();
  int num_threads = active_threads.load(cpp::MemoryOrder::RELAXED);
  result.cycles = cycles_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  result.standard_deviation =
      standard_deviation_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  result.min = min.load(cpp::MemoryOrder::RELAXED);
  result.max = max.load(cpp::MemoryOrder::RELAXED);
  result.samples = samples_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  result.total_iterations =
      iterations_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  result.total_time = time_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  gpu::memory_fence();
  log << GREEN << "[ RUN      ] " << RESET << b->get_name() << '\n';
  log << GREEN << "[       OK ] " << RESET << b->get_name() << ": "
      << result.cycles << " cycles, " << result.min << " min, " << result.max
      << " max, " << result.total_iterations << " iterations, "
      << result.total_time << " ns, "
      << static_cast<long>(result.standard_deviation)
      << " stddev (num threads: " << num_threads << ")\n";
}

void Benchmark::run_benchmarks() {
  uint64_t id = gpu::get_thread_id();
  gpu::sync_threads();

  for (Benchmark *b : benchmarks) {
    gpu::memory_fence();
    if (id == 0) {
      active_threads.store(0, cpp::MemoryOrder::RELAXED);
      cycles_sum.store(0, cpp::MemoryOrder::RELAXED);
      standard_deviation_sum.store(0, cpp::MemoryOrder::RELAXED);
      min.store(UINT64_MAX, cpp::MemoryOrder::RELAXED);
      max.store(0, cpp::MemoryOrder::RELAXED);
      samples_sum.store(0, cpp::MemoryOrder::RELAXED);
      iterations_sum.store(0, cpp::MemoryOrder::RELAXED);
      time_sum.store(0, cpp::MemoryOrder::RELAXED);
    }
    gpu::memory_fence();
    gpu::sync_threads();

    auto current_result = b->run();
    update_sums(current_result, active_threads, cycles_sum,
                standard_deviation_sum, min, max, samples_sum, iterations_sum,
                time_sum);
    gpu::sync_threads();

    if (id == 0) {
      print_results(b);
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
  uint64_t cycles_squared = 0;
  uint64_t min = UINT64_MAX;
  uint64_t max = 0;

  uint64_t overhead = UINT64_MAX;
  int overhead_iterations = 10;
  for (int i = 0; i < overhead_iterations; i++)
    overhead = cpp::min(overhead, LIBC_NAMESPACE::overhead());

  for (int64_t time_budget = options.max_duration; time_budget >= 0;) {
    uint64_t sample_cycles = 0;
    const clock_t start = static_cast<double>(clock());
    for (uint32_t i = 0; i < iterations; i++) {
      auto wrapper_intermediate = wrapper_func();
      uint64_t current_result = wrapper_intermediate - overhead;
      max = cpp::max(max, current_result);
      min = cpp::min(min, current_result);
      sample_cycles += current_result;
    }
    const clock_t end = clock();
    const clock_t duration_ns =
        ((end - start) * 1000 * 1000 * 1000) / CLOCKS_PER_SEC;
    total_time += duration_ns;
    time_budget -= duration_ns;
    samples++;
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
} // namespace LIBC_NAMESPACE_DECL
