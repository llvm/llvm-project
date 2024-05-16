#ifndef LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H
#define LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H

#include "benchmarks/gpu/BenchmarkLogger.h"
#include "benchmarks/gpu/timing/timing.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/CPP/string_view.h"

#include <stddef.h>
#include <stdint.h>

namespace LIBC_NAMESPACE {

namespace libc_gpu_benchmarks {

struct BenchmarkOptions {
  uint32_t initial_iterations = 1;
  uint32_t max_iterations = 10000000;
  uint32_t min_samples = 4;
  uint32_t max_samples = 1000;
  double epsilon = 0.01;
  double scaling_factor = 1.4;
};

struct Measurement {
  size_t iterations = 0;
  uint64_t elapsed_cycles = 0;
};

class RefinableRuntimeEstimation {
  uint64_t total_cycles = 0;
  size_t total_iterations = 0;

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
    double ratio = ((double)current_estimation / new_estimation) - 1.0;

    // Get absolute value
    if (ratio < 0)
      ratio *= -1;

    current_estimation = new_estimation;
    return ratio;
  }
};

struct BenchmarkResult {
  uint64_t cycles = 0;
  size_t samples = 0;
  size_t total_iterations = 0;
};

BenchmarkResult benchmark(const BenchmarkOptions &options,
                          cpp::function<uint64_t(void)> wrapper_func);

class Benchmark {
  Benchmark *next = nullptr;

public:
  virtual ~Benchmark() {}
  virtual void set_up() {}
  virtual void tear_down() {}

  static int run_benchmarks();

protected:
  static void add_benchmark(Benchmark *);

private:
  virtual void run() = 0;
  virtual const cpp::string_view get_name() const = 0;

  static Benchmark *start;
  static Benchmark *end;
};

class WrapperBenchmark : public Benchmark {
  const cpp::function<uint64_t(void)> func;
  const cpp::string_view name;

public:
  WrapperBenchmark(cpp::function<uint64_t(void)> func, char const *name)
      : func(func), name(name) {
    add_benchmark(this);
  }

private:
  void run() override {
    BenchmarkOptions options;
    auto result = benchmark(options, func);
    constexpr auto GREEN = "\033[32m";
    constexpr auto RESET = "\033[0m";
    blog << GREEN << "[ RUN      ] " << RESET << name << '\n';
    blog << GREEN << "[       OK ] " << RESET << name << ": " << result.cycles
         << " cycles, " << result.total_iterations << " iterations\n";
  }
  const cpp::string_view get_name() const override { return name; }
};
} // namespace libc_gpu_benchmarks
} // namespace LIBC_NAMESPACE

#define BENCHMARK(SuiteName, TestName, Func)                                   \
  LIBC_NAMESPACE::libc_gpu_benchmarks::WrapperBenchmark                        \
      SuiteName##_##TestName##_Instance(Func, #SuiteName "." #TestName);

#endif
