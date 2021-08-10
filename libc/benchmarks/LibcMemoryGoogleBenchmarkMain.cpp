#include "LibcBenchmark.h"
#include "LibcMemoryBenchmark.h"
#include "MemorySizeDistributions.h"
#include "benchmark/benchmark.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

using llvm::Align;
using llvm::ArrayRef;
using llvm::Twine;
using llvm::libc_benchmarks::BzeroConfiguration;
using llvm::libc_benchmarks::ComparisonSetup;
using llvm::libc_benchmarks::CopySetup;
using llvm::libc_benchmarks::MemcmpConfiguration;
using llvm::libc_benchmarks::MemcpyConfiguration;
using llvm::libc_benchmarks::MemorySizeDistribution;
using llvm::libc_benchmarks::MemsetConfiguration;
using llvm::libc_benchmarks::OffsetDistribution;
using llvm::libc_benchmarks::SetSetup;

namespace __llvm_libc {

extern void *memcpy(void *__restrict, const void *__restrict, size_t);
extern void *memset(void *, int, size_t);
extern void bzero(void *, size_t);
extern int memcmp(const void *, const void *, size_t);

} // namespace __llvm_libc

// List of implementations to test.
static constexpr MemcpyConfiguration kMemcpyConfigurations[] = {
    {__llvm_libc::memcpy, "__llvm_libc::memcpy"}};

static constexpr MemcmpConfiguration kMemcmpConfigurations[] = {
    {__llvm_libc::memcmp, "__llvm_libc::memcmp"}};

static constexpr MemsetConfiguration kMemsetConfigurations[] = {
    {__llvm_libc::memset, "__llvm_libc::memset"}};

static constexpr BzeroConfiguration kBzeroConfigurations[] = {
    {__llvm_libc::bzero, "__llvm_libc::bzero"}};

// Alignment to use for when accessing the buffers.
static constexpr Align kBenchmarkAlignment = Align::Constant<1>();

static std::mt19937_64 &getGenerator() {
  static std::mt19937_64 Generator(
      std::chrono::system_clock::now().time_since_epoch().count());
  return Generator;
}

template <typename SetupType, typename ConfigurationType> struct Runner {
  Runner(benchmark::State &S, llvm::ArrayRef<ConfigurationType> Configurations)
      : State(S), Distribution(SetupType::getDistributions()[State.range(0)]),
        Probabilities(Distribution.Probabilities),
        SizeSampler(Probabilities.begin(), Probabilities.end()),
        OffsetSampler(Setup.BufferSize, Probabilities.size() - 1,
                      kBenchmarkAlignment),
        Configuration(Configurations[State.range(1)]) {
    for (auto &P : Setup.Parameters) {
      P.OffsetBytes = OffsetSampler(getGenerator());
      P.SizeBytes = SizeSampler(getGenerator());
      Setup.checkValid(P);
    }
  }

  ~Runner() {
    const size_t AvgBytesPerIteration = Setup.getBatchBytes() / Setup.BatchSize;
    const size_t TotalBytes = State.iterations() * AvgBytesPerIteration;
    State.SetBytesProcessed(TotalBytes);
    State.SetItemsProcessed(State.iterations());
    State.SetLabel((Twine(Configuration.Name) + "," + Distribution.Name).str());
    State.counters["bytes_per_cycle"] = benchmark::Counter(
        TotalBytes / benchmark::CPUInfo::Get().cycles_per_second,
        benchmark::Counter::kIsRate);
  }

  inline void runBatch() {
    for (const auto &P : Setup.Parameters)
      benchmark::DoNotOptimize(Setup.Call(P, Configuration.Function));
  }

  size_t getBatchSize() const { return Setup.BatchSize; }

private:
  SetupType Setup;
  benchmark::State &State;
  MemorySizeDistribution Distribution;
  ArrayRef<double> Probabilities;
  std::discrete_distribution<unsigned> SizeSampler;
  OffsetDistribution OffsetSampler;
  ConfigurationType Configuration;
};

#define BENCHMARK_MEMORY_FUNCTION(BM_NAME, SETUP, CONFIGURATION_TYPE,          \
                                  CONFIGURATION_ARRAY_REF)                     \
  void BM_NAME(benchmark::State &State) {                                      \
    Runner<SETUP, CONFIGURATION_TYPE> Setup(State, CONFIGURATION_ARRAY_REF);   \
    const size_t BatchSize = Setup.getBatchSize();                             \
    while (State.KeepRunningBatch(BatchSize))                                  \
      Setup.runBatch();                                                        \
  }                                                                            \
  BENCHMARK(BM_NAME)->Apply([](benchmark::internal::Benchmark *benchmark) {    \
    const int64_t DistributionSize = SETUP::getDistributions().size();         \
    const int64_t ConfigurationSize = CONFIGURATION_ARRAY_REF.size();          \
    for (int64_t DistIndex = 0; DistIndex < DistributionSize; ++DistIndex)     \
      for (int64_t ConfIndex = 0; ConfIndex < ConfigurationSize; ++ConfIndex)  \
        benchmark->Args({DistIndex, ConfIndex});                               \
  })

BENCHMARK_MEMORY_FUNCTION(BM_Memcpy, CopySetup, MemcpyConfiguration,
                          llvm::makeArrayRef(kMemcpyConfigurations));
BENCHMARK_MEMORY_FUNCTION(BM_Memcmp, ComparisonSetup, MemcmpConfiguration,
                          llvm::makeArrayRef(kMemcmpConfigurations));
BENCHMARK_MEMORY_FUNCTION(BM_Memset, SetSetup, MemsetConfiguration,
                          llvm::makeArrayRef(kMemsetConfigurations));
BENCHMARK_MEMORY_FUNCTION(BM_Bzero, SetSetup, BzeroConfiguration,
                          llvm::makeArrayRef(kBzeroConfigurations));
