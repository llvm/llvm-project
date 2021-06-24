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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

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

static constexpr int64_t KiB = 1024;
static constexpr int64_t ParameterStorageBytes = 4 * KiB;
static constexpr int64_t L1LeftAsideBytes = 1 * KiB;

struct ParameterType {
  unsigned OffsetBytes : 16; // max : 16 KiB - 1
  unsigned SizeBytes : 16;   // max : 16 KiB - 1
};

#if defined(LIBC_BENCHMARK_FUNCTION_MEMCPY)
struct Benchmark {
  static constexpr auto GetDistributions = &getMemcpySizeDistributions;
  static constexpr size_t BufferCount = 2;

  Benchmark(const size_t BufferSize)
      : SrcBuffer(BufferSize), DstBuffer(BufferSize) {}

  inline auto functor() {
    return [this](ParameterType P) {
      __llvm_libc::memcpy(DstBuffer + P.OffsetBytes, SrcBuffer + P.OffsetBytes,
                          P.SizeBytes);
      return DstBuffer[P.OffsetBytes];
    };
  }

  AlignedBuffer SrcBuffer;
  AlignedBuffer DstBuffer;
};
#elif defined(LIBC_BENCHMARK_FUNCTION_MEMSET)
struct Benchmark {
  static constexpr auto GetDistributions = &getMemsetSizeDistributions;
  static constexpr size_t BufferCount = 1;

  Benchmark(const size_t BufferSize) : DstBuffer(BufferSize) {}

  inline auto functor() {
    return [this](ParameterType P) {
      __llvm_libc::memset(DstBuffer + P.OffsetBytes, P.OffsetBytes & 0xFF,
                          P.SizeBytes);
      return DstBuffer[P.OffsetBytes];
    };
  }

  AlignedBuffer DstBuffer;
};
#elif defined(LIBC_BENCHMARK_FUNCTION_BZERO)
struct Benchmark {
  static constexpr auto GetDistributions = &getMemsetSizeDistributions;
  static constexpr size_t BufferCount = 1;

  Benchmark(const size_t BufferSize) : DstBuffer(BufferSize) {}

  inline auto functor() {
    return [this](ParameterType P) {
      __llvm_libc::bzero(DstBuffer + P.OffsetBytes, P.SizeBytes);
      return DstBuffer[P.OffsetBytes];
    };
  }

  AlignedBuffer DstBuffer;
};
#elif defined(LIBC_BENCHMARK_FUNCTION_MEMCMP)
struct Benchmark {
  static constexpr auto GetDistributions = &getMemcmpSizeDistributions;
  static constexpr size_t BufferCount = 2;

  Benchmark(const size_t BufferSize)
      : BufferA(BufferSize), BufferB(BufferSize) {
    // The memcmp buffers always compare equal.
    memset(BufferA.begin(), 0xF, BufferSize);
    memset(BufferB.begin(), 0xF, BufferSize);
  }

  inline auto functor() {
    return [this](ParameterType P) {
      return __llvm_libc::memcmp(BufferA + P.OffsetBytes,
                                 BufferB + P.OffsetBytes, P.SizeBytes);
    };
  }

  AlignedBuffer BufferA;
  AlignedBuffer BufferB;
};
#else
#error "Missing LIBC_BENCHMARK_FUNCTION_XXX definition"
#endif

struct Harness : Benchmark {
  Harness(const size_t BufferSize, size_t BatchParameterCount,
          std::function<unsigned()> SizeSampler,
          std::function<unsigned()> OffsetSampler)
      : Benchmark(BufferSize), BufferSize(BufferSize),
        Parameters(BatchParameterCount), SizeSampler(SizeSampler),
        OffsetSampler(OffsetSampler) {}

  CircularArrayRef<ParameterType> generateBatch(size_t Iterations) {
    for (auto &P : Parameters) {
      P.OffsetBytes = OffsetSampler();
      P.SizeBytes = SizeSampler();
      if (P.OffsetBytes + P.SizeBytes >= BufferSize)
        report_fatal_error("Call would result in buffer overflow");
    }
    return cycle(makeArrayRef(Parameters), Iterations);
  }

private:
  const size_t BufferSize;
  std::vector<ParameterType> Parameters;
  std::function<unsigned()> SizeSampler;
  std::function<unsigned()> OffsetSampler;
};

size_t getL1DataCacheSize() {
  const std::vector<CacheInfo> &CacheInfos = HostState::get().Caches;
  const auto IsL1DataCache = [](const CacheInfo &CI) {
    return CI.Type == "Data" && CI.Level == 1;
  };
  const auto CacheIt = find_if(CacheInfos, IsL1DataCache);
  if (CacheIt != CacheInfos.end())
    return CacheIt->Size;
  report_fatal_error("Unable to read L1 Cache Data Size");
}

struct MemfunctionBenchmark {
  MemfunctionBenchmark(int64_t L1Size = getL1DataCacheSize())
      : AvailableSize(L1Size - L1LeftAsideBytes - ParameterStorageBytes),
        BufferSize(AvailableSize / Benchmark::BufferCount),
        BatchParameterCount(BufferSize / sizeof(ParameterType)) {
    // Handling command line flags
    if (AvailableSize <= 0 || BufferSize <= 0 || BatchParameterCount < 100)
      report_fatal_error("Not enough L1 cache");

    if (!isPowerOfTwoOrZero(AlignedAccess))
      report_fatal_error(AlignedAccess.ArgStr +
                         Twine(" must be a power of two or zero"));

    const bool HasDistributionName = !SizeDistributionName.empty();
    if (SweepMode && HasDistributionName)
      report_fatal_error("Select only one of `--" + Twine(SweepMode.ArgStr) +
                         "` or `--" + Twine(SizeDistributionName.ArgStr) + "`");

    if (SweepMode) {
      MaxSizeValue = SweepMaxSize;
    } else {
      std::map<StringRef, MemorySizeDistribution> Map;
      for (MemorySizeDistribution Distribution : Benchmark::GetDistributions())
        Map[Distribution.Name] = Distribution;
      if (Map.count(SizeDistributionName) == 0) {
        std::string Message;
        raw_string_ostream Stream(Message);
        Stream << "Unknown --" << SizeDistributionName.ArgStr << "='"
               << SizeDistributionName << "', available distributions:\n";
        for (const auto &Pair : Map)
          Stream << "'" << Pair.first << "'\n";
        report_fatal_error(Stream.str());
      }
      SizeDistribution = Map[SizeDistributionName];
      MaxSizeValue = SizeDistribution.Probabilities.size() - 1;
    }

    // Setup study.
    Study.StudyName = StudyName;
    Runtime &RI = Study.Runtime;
    RI.Host = HostState::get();
    RI.BufferSize = BufferSize;
    RI.BatchParameterCount = BatchParameterCount;

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
    if (SweepMode)
      SC.SweepModeMaxSize = SweepMaxSize;
    else
      SC.SizeDistributionName = SizeDistributionName;
    SC.AccessAlignment = MaybeAlign(AlignedAccess);
    SC.Function = LIBC_BENCHMARK_FUNCTION_NAME;
  }

  Study run() {
    if (SweepMode)
      runSweepMode();
    else
      runDistributionMode();
    return Study;
  }

private:
  const int64_t AvailableSize;
  const int64_t BufferSize;
  const size_t BatchParameterCount;
  size_t MaxSizeValue = 0;
  MemorySizeDistribution SizeDistribution;
  Study Study;
  std::mt19937_64 Gen;

  static constexpr bool isPowerOfTwoOrZero(size_t Value) {
    return (Value & (Value - 1U)) == 0;
  }

  std::function<unsigned()> geOffsetSampler() {
    return [this]() {
      static OffsetDistribution OD(BufferSize, MaxSizeValue,
                                   Study.Configuration.AccessAlignment);
      return OD(Gen);
    };
  }

  std::function<unsigned()> getSizeSampler() {
    return [this]() {
      static std::discrete_distribution<unsigned> Distribution(
          SizeDistribution.Probabilities.begin(),
          SizeDistribution.Probabilities.end());
      return Distribution(Gen);
    };
  }

  void reportProgress() {
    static size_t LastPercent = -1;
    const size_t TotalSteps = Study.Measurements.capacity();
    const size_t Steps = Study.Measurements.size();
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

  void runTrials(const BenchmarkOptions &Options,
                 std::function<unsigned()> SizeSampler,
                 std::function<unsigned()> OffsetSampler) {
    Harness B(BufferSize, BatchParameterCount, SizeSampler, OffsetSampler);
    for (size_t i = 0; i < NumTrials; ++i) {
      const BenchmarkResult Result = benchmark(Options, B, B.functor());
      Study.Measurements.push_back(Result.BestGuess);
      reportProgress();
    }
  }

  void runSweepMode() {
    Study.Measurements.reserve(NumTrials * SweepMaxSize);

    BenchmarkOptions &BO = Study.Runtime.BenchmarkOptions;
    BO.MinDuration = std::chrono::milliseconds(1);
    BO.InitialIterations = 100;

    for (size_t Size = 0; Size <= SweepMaxSize; ++Size) {
      const auto SizeSampler = [Size]() { return Size; };
      runTrials(BO, SizeSampler, geOffsetSampler());
    }
  }

  void runDistributionMode() {
    Study.Measurements.reserve(NumTrials);

    BenchmarkOptions &BO = Study.Runtime.BenchmarkOptions;
    BO.MinDuration = std::chrono::milliseconds(10);
    BO.InitialIterations = BatchParameterCount * 10;

    runTrials(BO, getSizeSampler(), geOffsetSampler());
  }
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
  MemfunctionBenchmark MB;
  writeStudy(MB.run());
}

} // namespace libc_benchmarks
} // namespace llvm

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
#ifndef NDEBUG
  static_assert(
      false,
      "For reproducibility benchmarks should not be compiled in DEBUG mode.");
#endif
  llvm::libc_benchmarks::main();
  return EXIT_SUCCESS;
}
