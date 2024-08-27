#include "LibcGpuBenchmark.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/string.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/fixedvector.h"
#include "src/__support/macros/config.h"
#include "src/stdio/printf.h"
#include "src/stdlib/srand.h"
#include "src/time/gpu/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace benchmarks {

FixedVector<Benchmark *, 64> benchmarks;

void Benchmark::add_benchmark(Benchmark *benchmark) {
  benchmarks.push_back(benchmark);
}

struct AtomicBenchmarkSums {
  cpp::Atomic<uint64_t> cycles_sum = 0;
  cpp::Atomic<uint64_t> standard_deviation_sum = 0;
  cpp::Atomic<uint64_t> min = UINT64_MAX;
  cpp::Atomic<uint64_t> max = 0;
  cpp::Atomic<uint32_t> samples_sum = 0;
  cpp::Atomic<uint32_t> iterations_sum = 0;
  cpp::Atomic<clock_t> time_sum = 0;
  cpp::Atomic<uint64_t> active_threads = 0;

  void reset() {
    cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    active_threads.store(0, cpp::MemoryOrder::RELAXED);
    cycles_sum.store(0, cpp::MemoryOrder::RELAXED);
    standard_deviation_sum.store(0, cpp::MemoryOrder::RELAXED);
    min.store(UINT64_MAX, cpp::MemoryOrder::RELAXED);
    max.store(0, cpp::MemoryOrder::RELAXED);
    samples_sum.store(0, cpp::MemoryOrder::RELAXED);
    iterations_sum.store(0, cpp::MemoryOrder::RELAXED);
    time_sum.store(0, cpp::MemoryOrder::RELAXED);
    cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
  }

  void update(const BenchmarkResult &result) {
    cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    active_threads.fetch_add(1, cpp::MemoryOrder::RELAXED);

    cycles_sum.fetch_add(result.cycles, cpp::MemoryOrder::RELAXED);
    standard_deviation_sum.fetch_add(
        static_cast<uint64_t>(result.standard_deviation),
        cpp::MemoryOrder::RELAXED);

    // Perform a CAS loop to atomically update the min
    uint64_t orig_min = min.load(cpp::MemoryOrder::RELAXED);
    while (!min.compare_exchange_strong(
        orig_min, cpp::min(orig_min, result.min), cpp::MemoryOrder::ACQUIRE,
        cpp::MemoryOrder::RELAXED))
      ;

    // Perform a CAS loop to atomically update the max
    uint64_t orig_max = max.load(cpp::MemoryOrder::RELAXED);
    while (!max.compare_exchange_strong(
        orig_max, cpp::max(orig_max, result.max), cpp::MemoryOrder::ACQUIRE,
        cpp::MemoryOrder::RELAXED))
      ;

    samples_sum.fetch_add(result.samples, cpp::MemoryOrder::RELAXED);
    iterations_sum.fetch_add(result.total_iterations,
                             cpp::MemoryOrder::RELAXED);
    time_sum.fetch_add(result.total_time, cpp::MemoryOrder::RELAXED);
    cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
  }
};

AtomicBenchmarkSums all_results;
constexpr auto GREEN = "\033[32m";
constexpr auto RESET = "\033[0m";

void print_results(Benchmark *b) {
  BenchmarkResult result;
  cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
  int num_threads = all_results.active_threads.load(cpp::MemoryOrder::RELAXED);
  result.cycles =
      all_results.cycles_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  result.standard_deviation =
      all_results.standard_deviation_sum.load(cpp::MemoryOrder::RELAXED) /
      num_threads;
  result.min = all_results.min.load(cpp::MemoryOrder::RELAXED);
  result.max = all_results.max.load(cpp::MemoryOrder::RELAXED);
  result.samples =
      all_results.samples_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  result.total_iterations =
      all_results.iterations_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  const uint64_t duration_ns =
      all_results.time_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  const uint64_t duration_us = duration_ns / 1000;
  const uint64_t duration_ms = duration_ns / (1000 * 1000);
  uint64_t converted_duration = duration_ns;
  const char *time_unit;
  if (duration_ms != 0) {
    converted_duration = duration_ms;
    time_unit = "ms";
  } else if (duration_us != 0) {
    converted_duration = duration_us;
    time_unit = "us";
  } else {
    converted_duration = duration_ns;
    time_unit = "ns";
  }
  result.total_time = converted_duration;
  // result.total_time =
  //     all_results.time_sum.load(cpp::MemoryOrder::RELAXED) / num_threads;
  cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);

  LIBC_NAMESPACE::printf(
      "%-20s |%8ld |%8ld |%8ld |%11d |%14ld %2s |%9ld |%9d |\n",
      b->get_test_name().data(), result.cycles, result.min, result.max,
      result.total_iterations, result.total_time, time_unit,
      static_cast<uint64_t>(result.standard_deviation), num_threads);
}

void print_header() {
  LIBC_NAMESPACE::printf("%s", GREEN);
  LIBC_NAMESPACE::printf("Running Suite: %-10s\n",
                         benchmarks[0]->get_suite_name().data());
  LIBC_NAMESPACE::printf("%s", RESET);
  cpp::string titles =
      "Benchmark            |  Cycles |     Min |     Max | "
      "Iterations | Time / Iteration |   Stddev |  Threads |\n";
  LIBC_NAMESPACE::printf(titles.data());

  cpp::string separator(titles.size(), '-');
  separator[titles.size() - 1] = '\n';
  LIBC_NAMESPACE::printf(separator.data());
}

void Benchmark::run_benchmarks() {
  uint64_t id = gpu::get_thread_id();

  if (id == 0) {
    print_header();
    LIBC_NAMESPACE::srand(gpu::processor_clock());
  }

  gpu::sync_threads();

  for (Benchmark *b : benchmarks) {
    if (id == 0)
      all_results.reset();

    gpu::sync_threads();
    if (b->num_threads == static_cast<uint32_t>(-1) || id < b->num_threads) {
      auto current_result = b->run();
      all_results.update(current_result);
    }
    gpu::sync_threads();

    if (id == 0)
      print_results(b);
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
        total_iterations >= options.min_iterations &&
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
  result.total_time = total_time / total_iterations;
  return result;
};

} // namespace benchmarks
} // namespace LIBC_NAMESPACE_DECL
