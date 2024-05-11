#ifndef LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H
#define LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H

#include "benchmarks/gpu/timing/timing.h"

#include "benchmarks/gpu/BenchmarkLogger.h"

#include <stddef.h>
#include <stdint.h>

namespace LIBC_NAMESPACE {

namespace libc_gpu_benchmarks {

struct BenchmarkOptions {
  uint32_t InitialIterations = 1;
  uint32_t MaxIterations = 10000000;
  uint32_t MinSamples = 4;
  uint32_t MaxSamples = 1000;
  double Epsilon = 0.01;
  double ScalingFactor = 1.4;
};

struct Measurement {
  size_t Iterations = 0;
  uint64_t ElapsedCycles = 0;
};

class RefinableRuntimeEstimation {
  uint64_t TotalCycles = 0;
  size_t TotalIterations = 0;

public:
  uint64_t Update(const Measurement &M) {
    TotalCycles += M.ElapsedCycles;
    TotalIterations += M.Iterations;
    return TotalCycles / TotalIterations;
  }
};

// Tracks the progression of the runtime estimation
class RuntimeEstimationProgression {
  RefinableRuntimeEstimation RRE;

public:
  uint64_t CurrentEstimation = 0;

  double ComputeImprovement(const Measurement &M) {
    const uint64_t NewEstimation = RRE.Update(M);
    double Ratio = ((double)CurrentEstimation / NewEstimation) - 1.0;

    // Get absolute value
    if (Ratio < 0) {
      Ratio *= -1;
    }

    CurrentEstimation = NewEstimation;
    return Ratio;
  }
};

struct BenchmarkResult {
  uint64_t Cycles = 0;
  size_t Samples = 0;
  size_t TotalIterations = 0;
};

BenchmarkResult benchmark(const BenchmarkOptions &Options,
                          uint64_t (*WrapperFunc)());

class Benchmark {
  Benchmark *Next = nullptr;

public:
  virtual ~Benchmark() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  static int runBenchmarks();

protected:
  static void addBenchmark(Benchmark *);

private:
  virtual void Run() = 0;
  virtual const char *getName() const = 0;

  static Benchmark *Start;
  static Benchmark *End;
};

class WrapperBenchmark : public Benchmark {
  using BenchmarkWrapperFunction = uint64_t (*)();
  BenchmarkWrapperFunction Func;
  const char *Name;

public:
  WrapperBenchmark(BenchmarkWrapperFunction Func, char const *Name)
      : Func(Func), Name(Name) {
    addBenchmark(this);
  }

private:
  void Run() override {
    BenchmarkOptions Options;
    auto result = benchmark(Options, Func);
    constexpr auto GREEN = "\033[32m";
    constexpr auto RESET = "\033[0m";
    blog << GREEN << "[ RUN      ] " << RESET << Name << '\n';
    blog << GREEN << "[       OK ] " << RESET << Name << ": " << result.Cycles
         << " cycles, " << result.TotalIterations << " iterations\n";
  }
  const char *getName() const override { return Name; }
};
} // namespace libc_gpu_benchmarks
} // namespace LIBC_NAMESPACE

#define BENCHMARK(SuiteName, TestName, Func)                                   \
  LIBC_NAMESPACE::libc_gpu_benchmarks::WrapperBenchmark                        \
      SuiteName##_##TestName##_Instance(Func, #SuiteName "." #TestName);

#endif
