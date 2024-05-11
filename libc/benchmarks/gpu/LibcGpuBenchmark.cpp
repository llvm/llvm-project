#include "LibcGpuBenchmark.h"

namespace LIBC_NAMESPACE {
namespace libc_gpu_benchmarks {

Benchmark *Benchmark::Start = nullptr;
Benchmark *Benchmark::End = nullptr;

void Benchmark::addBenchmark(Benchmark *B) {
  if (End == nullptr) {
    Start = B;
    End = B;
    return;
  }

  End->Next = B;
  End = B;
}

int Benchmark::runBenchmarks() {
  for (Benchmark *B = Start; B != nullptr; B = B->Next) {
    B->Run();
  }

  return 0;
}

BenchmarkResult benchmark(const BenchmarkOptions &Options,
                          uint64_t (*WrapperFunc)()) {
  BenchmarkResult Result;
  RuntimeEstimationProgression REP;
  size_t TotalIterations = 0;
  size_t Iterations = Options.InitialIterations;
  if (Iterations < (uint32_t)1) {
    Iterations = 1;
  }
  size_t Samples = 0;
  uint64_t BestGuess = 0;
  uint64_t TotalCycles = 0;
  for (;;) {
    uint64_t SampleCycles = 0;
    for (uint32_t i = 0; i < Iterations; i++) {
      auto overhead = LIBC_NAMESPACE::overhead();
      uint64_t result = WrapperFunc() - overhead;
      SampleCycles += result;
    }

    Samples++;
    TotalCycles += SampleCycles;
    TotalIterations += Iterations;
    const double ChangeRatio =
        REP.ComputeImprovement({Iterations, SampleCycles});
    BestGuess = REP.CurrentEstimation;

    if (Samples >= Options.MaxSamples || Iterations >= Options.MaxIterations) {
      break;
    } else if (Samples >= Options.MinSamples && ChangeRatio < Options.Epsilon) {
      break;
    }

    Iterations *= Options.ScalingFactor;
  }
  Result.Cycles = BestGuess;
  Result.Samples = Samples;
  Result.TotalIterations = TotalIterations;
  return Result;
};

} // namespace libc_gpu_benchmarks
} // namespace LIBC_NAMESPACE
