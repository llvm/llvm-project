#include "LibcGpuBenchmark.h"

namespace LIBC_NAMESPACE {
namespace libc_gpu_benchmarks {

Benchmark *Benchmark::start = nullptr;
Benchmark *Benchmark::end = nullptr;

void Benchmark::add_benchmark(Benchmark *benchmark) {
  if (end == nullptr) {
    start = benchmark;
    end = benchmark;
    return;
  }
  end->next = benchmark;
  end = benchmark;
}

int Benchmark::run_benchmarks() {
  for (Benchmark *b = start; b != nullptr; b = b->next)
    b->run();
  return 0;
}

BenchmarkResult benchmark(const BenchmarkOptions &options,
                          cpp::function<uint64_t(void)> wrapper_func) {
  BenchmarkResult result;
  RuntimeEstimationProgression rep;
  size_t total_iterations = 0;
  size_t iterations = options.initial_iterations;
  if (iterations < (uint32_t)1)
    iterations = 1;

  size_t samples = 0;
  uint64_t best_guess = 0;
  uint64_t total_cycles = 0;
  for (;;) {
    uint64_t sample_cycles = 0;
    uint64_t overhead = LIBC_NAMESPACE::overhead();
    for (uint32_t i = 0; i < iterations; i++) {
      uint64_t result = wrapper_func() - overhead;
      sample_cycles += result;
    }

    samples++;
    total_cycles += sample_cycles;
    total_iterations += iterations;
    const double change_ratio =
        rep.compute_improvement({iterations, sample_cycles});
    best_guess = rep.current_estimation;

    if (samples >= options.max_samples ||
        iterations >= options.max_iterations) {
      break;
    } else if (samples >= options.min_samples &&
               change_ratio < options.epsilon) {
      break;
    }

    iterations *= options.scaling_factor;
  }
  result.cycles = best_guess;
  result.samples = samples;
  result.total_iterations = total_iterations;
  return result;
};

} // namespace libc_gpu_benchmarks
} // namespace LIBC_NAMESPACE
