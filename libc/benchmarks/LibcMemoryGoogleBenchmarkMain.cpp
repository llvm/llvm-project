#include "LibcBenchmark.h"
#include "LibcMemoryBenchmark.h"
#include "MemorySizeDistributions.h"
#include "benchmark/benchmark.h"
#include "llvm/ADT/ArrayRef.h"
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
#include "llvm/ADT/Twine.h"
#endif
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

using llvm::Align;
using llvm::ArrayRef;
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
using llvm::Twine;
#endif
using llvm::libc_benchmarks::BzeroConfiguration;
using llvm::libc_benchmarks::ComparisonSetup;
using llvm::libc_benchmarks::CopySetup;
using llvm::libc_benchmarks::MemcmpOrBcmpConfiguration;
using llvm::libc_benchmarks::MemcpyConfiguration;
using llvm::libc_benchmarks::MemmoveConfiguration;
using llvm::libc_benchmarks::MemorySizeDistribution;
using llvm::libc_benchmarks::MemsetConfiguration;
using llvm::libc_benchmarks::MoveSetup;
using llvm::libc_benchmarks::OffsetDistribution;
using llvm::libc_benchmarks::SetSetup;

// Alignment to use for when accessing the buffers.
static constexpr Align kBenchmarkAlignment = Align::Constant<1>();

static std::mt19937_64 &getGenerator() {
  static std::mt19937_64 Generator(
      std::chrono::system_clock::now().time_since_epoch().count());
  return Generator;
}

template <typename SetupType, typename ConfigurationType> struct Runner {
  Runner(benchmark::State &S, llvm::ArrayRef<ConfigurationType> Configurations)
      : State(S), distribution(SetupType::get_distributions()[State.range(0)]),
        probabilities(distribution.probabilities),
        size_sampler(probabilities.begin(), probabilities.end()),
        offset_sampler(Setup.buffer_size, probabilities.size() - 1,
                       kBenchmarkAlignment),
        configuration(Configurations[State.range(1)]) {
    for (auto &p : Setup.parameters) {
      p.offset_bytes = offset_sampler(getGenerator());
      p.size_bytes = size_sampler(getGenerator());
      Setup.check_valid(p);
    }
  }

  ~Runner() {
    const size_t total_bytes =
        (State.iterations() * Setup.get_batch_bytes()) / Setup.batch_size;
    State.SetBytesProcessed(total_bytes);
    State.SetItemsProcessed(State.iterations());
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
    State.SetLabel((Twine(configuration.name) + "," + distribution.name).str());
#else
    State.SetLabel(configuration.name.str() + "," + distribution.name.str());
#endif
    State.counters["bytes_per_cycle"] = benchmark::Counter(
        total_bytes / benchmark::CPUInfo::Get().cycles_per_second,
        benchmark::Counter::kIsRate);
  }

  inline void run_batch() {
    for (const auto &p : Setup.parameters)
      benchmark::DoNotOptimize(Setup.call(p, configuration.function));
  }

  size_t get_batch_size() const { return Setup.batch_size; }

private:
  SetupType Setup;
  benchmark::State &State;
  MemorySizeDistribution distribution;
  ArrayRef<double> probabilities;
  std::discrete_distribution<unsigned> size_sampler;
  OffsetDistribution offset_sampler;
  ConfigurationType configuration;
};

#define BENCHMARK_MEMORY_FUNCTION(BM_NAME, SETUP, CONFIGURATION_TYPE,          \
                                  CONFIGURATION_ARRAY_REF)                     \
  void BM_NAME(benchmark::State &State) {                                      \
    Runner<SETUP, CONFIGURATION_TYPE> Setup(State, CONFIGURATION_ARRAY_REF);   \
    const size_t batch_size_val = Setup.get_batch_size();                      \
    while (State.KeepRunningBatch(batch_size_val))                             \
      Setup.run_batch();                                                       \
  }                                                                            \
  BENCHMARK(BM_NAME)->Apply([](benchmark::internal::Benchmark *benchmark) {    \
    const int64_t DistributionSize = SETUP::get_distributions().size();        \
    const int64_t ConfigurationSize = CONFIGURATION_ARRAY_REF.size();          \
    for (int64_t DistIndex = 0; DistIndex < DistributionSize; ++DistIndex)     \
      for (int64_t ConfIndex = 0; ConfIndex < ConfigurationSize; ++ConfIndex)  \
        benchmark->Args({DistIndex, ConfIndex});                               \
  })

extern llvm::ArrayRef<MemcpyConfiguration> getMemcpyConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Memcpy, CopySetup, MemcpyConfiguration,
                          getMemcpyConfigurations());

extern llvm::ArrayRef<MemmoveConfiguration> getMemmoveConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Memmove, MoveSetup, MemmoveConfiguration,
                          getMemmoveConfigurations());

extern llvm::ArrayRef<MemcmpOrBcmpConfiguration> getMemcmpConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Memcmp, ComparisonSetup, MemcmpOrBcmpConfiguration,
                          getMemcmpConfigurations());

extern llvm::ArrayRef<MemcmpOrBcmpConfiguration> getBcmpConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Bcmp, ComparisonSetup, MemcmpOrBcmpConfiguration,
                          getBcmpConfigurations());

extern llvm::ArrayRef<MemsetConfiguration> getMemsetConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Memset, SetSetup, MemsetConfiguration,
                          getMemsetConfigurations());

extern llvm::ArrayRef<BzeroConfiguration> getBzeroConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Bzero, SetSetup, BzeroConfiguration,
                          getBzeroConfigurations());
