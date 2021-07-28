//===-- Benchmark ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSON.h"
#include "LibcBenchmark.h"
#include "LibcMemoryBenchmark.h"
#include "MemorySizeDistributions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <unistd.h>

namespace __llvm_libc {

extern void *memcpy(void *__restrict, const void *__restrict, size_t);
extern void *memset(void *, int, size_t);
extern void bzero(void *, size_t);
extern int memcmp(const void *, const void *, size_t);

} // namespace __llvm_libc

namespace llvm {
namespace libc_benchmarks {

static cl::opt<std::string>
    StudyName("study-name", cl::desc("The name for this study"), cl::Required);

static cl::opt<std::string>
    SizeDistributionName("size-distribution-name",
                         cl::desc("The name of the distribution to use"));

static cl::opt<bool>
    SweepMode("sweep-mode",
              cl::desc("If set, benchmark all sizes from 0 to sweep-max-size"));

static cl::opt<uint32_t>
    SweepMaxSize("sweep-max-size",
                 cl::desc("The maximum size to use in sweep-mode"),
                 cl::init(256));

static cl::opt<uint32_t>
    AlignedAccess("aligned-access",
                  cl::desc("The alignment to use when accessing the buffers\n"
                           "Default is unaligned\n"
                           "Use 0 to disable address randomization"),
                  cl::init(1));

static cl::opt<std::string> Output("output",
                                   cl::desc("Specify output filename"),
                                   cl::value_desc("filename"), cl::init("-"));

static cl::opt<uint32_t>
    NumTrials("num-trials", cl::desc("The number of benchmarks run to perform"),
              cl::init(1));

#if defined(LIBC_BENCHMARK_FUNCTION_MEMCPY)
#define LIBC_BENCHMARK_FUNCTION LIBC_BENCHMARK_FUNCTION_MEMCPY
using BenchmarkHarness = CopyHarness;
#elif defined(LIBC_BENCHMARK_FUNCTION_MEMSET)
#define LIBC_BENCHMARK_FUNCTION LIBC_BENCHMARK_FUNCTION_MEMSET
using BenchmarkHarness = SetHarness;
#elif defined(LIBC_BENCHMARK_FUNCTION_BZERO)
#define LIBC_BENCHMARK_FUNCTION LIBC_BENCHMARK_FUNCTION_BZERO
using BenchmarkHarness = SetHarness;
#elif defined(LIBC_BENCHMARK_FUNCTION_MEMCMP)
#define LIBC_BENCHMARK_FUNCTION LIBC_BENCHMARK_FUNCTION_MEMCMP
using BenchmarkHarness = ComparisonHarness;
#else
#error "Missing LIBC_BENCHMARK_FUNCTION_XXX definition"
#endif

struct MemfunctionBenchmarkBase : public BenchmarkHarness {
  MemfunctionBenchmarkBase() : ReportProgress(isatty(fileno(stdout))) {}
  virtual ~MemfunctionBenchmarkBase() {}

  virtual Study run() = 0;

  CircularArrayRef<ParameterBatch::ParameterType>
  generateBatch(size_t Iterations) {
    randomize();
    return cycle(makeArrayRef(Parameters), Iterations);
  }

protected:
  Study createStudy() {
    Study Study;
    // Harness study.
    Study.StudyName = StudyName;
    Runtime &RI = Study.Runtime;
    RI.Host = HostState::get();
    RI.BufferSize = BufferSize;
    RI.BatchParameterCount = BatchSize;

    BenchmarkOptions &BO = RI.BenchmarkOptions;
    BO.MinDuration = std::chrono::milliseconds(1);
    BO.MaxDuration = std::chrono::seconds(1);
    BO.MaxIterations = 10'000'000U;
    BO.MinSamples = 4;
    BO.MaxSamples = 1000;
    BO.Epsilon = 0.01; // 1%
    BO.ScalingFactor = 1.4;

    StudyConfiguration &SC = Study.Configuration;
    SC.NumTrials = NumTrials;
    SC.IsSweepMode = SweepMode;
    SC.AccessAlignment = MaybeAlign(AlignedAccess);
    SC.Function = LIBC_BENCHMARK_FUNCTION_NAME;
    return Study;
  }

  void runTrials(const BenchmarkOptions &Options,
                 std::vector<Duration> &Measurements) {
    for (size_t i = 0; i < NumTrials; ++i) {
      const BenchmarkResult Result = benchmark(
          Options, *this, [this](ParameterBatch::ParameterType Parameter) {
            return Call(Parameter, LIBC_BENCHMARK_FUNCTION);
          });
      Measurements.push_back(Result.BestGuess);
      reportProgress(Measurements);
    }
  }

  virtual void randomize() = 0;

private:
  bool ReportProgress;

  void reportProgress(const std::vector<Duration> &Measurements) {
    if (!ReportProgress)
      return;
    static size_t LastPercent = -1;
    const size_t TotalSteps = Measurements.capacity();
    const size_t Steps = Measurements.size();
    const size_t Percent = 100 * Steps / TotalSteps;
    if (Percent == LastPercent)
      return;
    LastPercent = Percent;
    size_t I = 0;
    errs() << '[';
    for (; I <= Percent; ++I)
      errs() << '#';
    for (; I <= 100; ++I)
      errs() << '_';
    errs() << "] " << Percent << '%' << '\r';
  }
};

struct MemfunctionBenchmarkSweep final : public MemfunctionBenchmarkBase {
  MemfunctionBenchmarkSweep()
      : OffsetSampler(MemfunctionBenchmarkBase::BufferSize, SweepMaxSize,
                      MaybeAlign(AlignedAccess)) {}

  virtual void randomize() override {
    for (auto &P : Parameters) {
      P.OffsetBytes = OffsetSampler(Gen);
      P.SizeBytes = CurrentSweepSize;
      checkValid(P);
    }
  }

  virtual Study run() override {
    Study Study = createStudy();
    Study.Configuration.SweepModeMaxSize = SweepMaxSize;
    BenchmarkOptions &BO = Study.Runtime.BenchmarkOptions;
    BO.MinDuration = std::chrono::milliseconds(1);
    BO.InitialIterations = 100;
    auto &Measurements = Study.Measurements;
    Measurements.reserve(NumTrials * SweepMaxSize);
    for (size_t Size = 0; Size <= SweepMaxSize; ++Size) {
      CurrentSweepSize = Size;
      runTrials(BO, Measurements);
    }
    return Study;
  }

private:
  size_t CurrentSweepSize = 0;
  OffsetDistribution OffsetSampler;
  std::mt19937_64 Gen;
};

struct MemfunctionBenchmarkDistribution final
    : public MemfunctionBenchmarkBase {
  MemfunctionBenchmarkDistribution(MemorySizeDistribution Distribution)
      : Distribution(Distribution), Probabilities(Distribution.Probabilities),
        SizeSampler(Probabilities.begin(), Probabilities.end()),
        OffsetSampler(MemfunctionBenchmarkBase::BufferSize,
                      Probabilities.size() - 1, MaybeAlign(AlignedAccess)) {}

  virtual void randomize() override {
    for (auto &P : Parameters) {
      P.OffsetBytes = OffsetSampler(Gen);
      P.SizeBytes = SizeSampler(Gen);
      checkValid(P);
    }
  }

  virtual Study run() override {
    Study Study = createStudy();
    Study.Configuration.SizeDistributionName = Distribution.Name.str();
    BenchmarkOptions &BO = Study.Runtime.BenchmarkOptions;
    BO.MinDuration = std::chrono::milliseconds(10);
    BO.InitialIterations = BatchSize * 10;
    auto &Measurements = Study.Measurements;
    Measurements.reserve(NumTrials);
    runTrials(BO, Measurements);
    return Study;
  }

private:
  MemorySizeDistribution Distribution;
  ArrayRef<double> Probabilities;
  std::discrete_distribution<unsigned> SizeSampler;
  OffsetDistribution OffsetSampler;
  std::mt19937_64 Gen;
};

void writeStudy(const Study &S) {
  std::error_code EC;
  raw_fd_ostream FOS(Output, EC);
  if (EC)
    report_fatal_error(Twine("Could not open file: ")
                           .concat(EC.message())
                           .concat(", ")
                           .concat(Output));
  json::OStream JOS(FOS);
  serializeToJson(S, JOS);
  FOS << "\n";
}

void main() {
  checkRequirements();
  if (!isPowerOf2_32(AlignedAccess))
    report_fatal_error(AlignedAccess.ArgStr +
                       Twine(" must be a power of two or zero"));

  const bool HasDistributionName = !SizeDistributionName.empty();
  if (SweepMode && HasDistributionName)
    report_fatal_error("Select only one of `--" + Twine(SweepMode.ArgStr) +
                       "` or `--" + Twine(SizeDistributionName.ArgStr) + "`");

  std::unique_ptr<MemfunctionBenchmarkBase> Benchmark;
  if (SweepMode)
    Benchmark.reset(new MemfunctionBenchmarkSweep());
  else
    Benchmark.reset(new MemfunctionBenchmarkDistribution(getDistributionOrDie(
        BenchmarkHarness::Distributions, SizeDistributionName)));
  writeStudy(Benchmark->run());
}

} // namespace libc_benchmarks
} // namespace llvm

#ifndef NDEBUG
#error For reproducibility benchmarks should not be compiled in DEBUG mode.
#endif

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  llvm::libc_benchmarks::main();
  return EXIT_SUCCESS;
}
