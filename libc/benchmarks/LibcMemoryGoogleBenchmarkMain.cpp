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
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using llvm::Align;
using llvm::ArrayRef;
#ifdef LIBC_BENCHMARKS_HAS_LLVM_SUPPORT
using llvm::Twine;
#endif
using llvm::libc_benchmarks::AlignedBuffer;
using llvm::libc_benchmarks::BzeroConfiguration;
using llvm::libc_benchmarks::ComparisonSetup;
using llvm::libc_benchmarks::CopySetup;
using llvm::libc_benchmarks::MemchrConfiguration;
using llvm::libc_benchmarks::MemcmpOrBcmpConfiguration;
using llvm::libc_benchmarks::MemcmpOrBcmpFunction;
using llvm::libc_benchmarks::MemcpyConfiguration;
using llvm::libc_benchmarks::MemmoveConfiguration;
using llvm::libc_benchmarks::MemorySizeDistribution;
using llvm::libc_benchmarks::MemsetConfiguration;
using llvm::libc_benchmarks::MoveSetup;
using llvm::libc_benchmarks::OffsetDistribution;
using llvm::libc_benchmarks::SearchSetup;
using llvm::libc_benchmarks::SetSetup;
using llvm::libc_benchmarks::StrcmpConfiguration;
using llvm::libc_benchmarks::StrlenConfiguration;

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

extern llvm::ArrayRef<MemchrConfiguration> getMemchrConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Memchr, SearchSetup, MemchrConfiguration,
                          getMemchrConfigurations());

extern llvm::ArrayRef<StrlenConfiguration> getStrlenConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Strlen, SearchSetup, StrlenConfiguration,
                          getStrlenConfigurations());

extern llvm::ArrayRef<StrcmpConfiguration> getStrcmpConfigurations();
BENCHMARK_MEMORY_FUNCTION(BM_Strcmp, ComparisonSetup, StrcmpConfiguration,
                          getStrcmpConfigurations());

namespace {

// Large enough that modern branch predictors cannot memorize the repeating
// sequence of comparison sizes within a batch.
constexpr size_t kComparisonBatchSize = 64 * 1024;

// Compact 8-byte descriptor (all benchmark sizes and buffer offsets fit in
// uint16_t) for power-of-two indexing and smaller cache footprint.
struct ComparisonPair {
  uint16_t lhs_offset;
  uint16_t rhs_offset;
  uint16_t size;
  uint16_t pad;
};

template <bool IsLatency>
[[gnu::noinline]] void
runComparisonLoop(benchmark::State &state, const char *lhs, const char *rhs,
                  ArrayRef<ComparisonPair> pairs, MemcmpOrBcmpFunction func) {
  const size_t n = pairs.size();
  while (state.KeepRunningBatch(n)) {
    if constexpr (IsLatency) {
      size_t count = 0;
      for (size_t i = 0; i < n; ++count) {
        const ComparisonPair &p = pairs[i];
        // Feed the zero/non-zero result into the next index branchlessly so
        // each comparison must resolve before the next begins, mirroring a
        // caller's branch on the return value without letting comparisons
        // overlap.
        i += 1 + static_cast<size_t>(
                     func(lhs + p.lhs_offset, rhs + p.rhs_offset, p.size) != 0);
      }
      if (count != n)
        std::abort();
    } else {
      for (const ComparisonPair &p : pairs) {
        // Every pair is equal, so this branch is predicted and successive
        // comparisons can overlap.
        if (__builtin_expect(
                func(lhs + p.lhs_offset, rhs + p.rhs_offset, p.size) != 0, 0))
          std::abort();
      }
    }
  }
}

template <bool IsLatency, typename SizeSampler>
void runComparisonBatch(benchmark::State &state,
                        const MemcmpOrBcmpConfiguration &config,
                        size_t max_size, SizeSampler &&size_sampler,
                        const std::string &label) {
  // Keep data buffers small enough to stay in L1 while varying alignment.
  const size_t buffer_size = max_size + 256;
  AlignedBuffer lhs_buffer(buffer_size);
  AlignedBuffer rhs_buffer(buffer_size);
  std::memset(lhs_buffer.begin(), 0x0F, buffer_size);
  std::memset(rhs_buffer.begin(), 0x0F, buffer_size);

  OffsetDistribution offset_sampler(buffer_size, max_size, kBenchmarkAlignment);
  auto &gen = getGenerator();
  std::vector<ComparisonPair> pairs;
  pairs.reserve(kComparisonBatchSize);
  uint64_t batch_bytes = 0;
  for (size_t i = 0; i < kComparisonBatchSize; ++i) {
    const size_t size = size_sampler(gen);
    pairs.push_back({static_cast<uint16_t>(offset_sampler(gen)),
                     static_cast<uint16_t>(offset_sampler(gen)),
                     static_cast<uint16_t>(size), 0});
    batch_bytes += size;
  }

  runComparisonLoop<IsLatency>(state, lhs_buffer.begin(), rhs_buffer.begin(),
                               pairs, config.function);

  const uint64_t total_bytes =
      (state.iterations() / kComparisonBatchSize) * batch_bytes;
  state.SetBytesProcessed(total_bytes);
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(label);
  state.counters["bytes_per_cycle"] = benchmark::Counter(
      total_bytes / benchmark::CPUInfo::Get().cycles_per_second,
      benchmark::Counter::kIsRate);
}

void runDistributionLatencyBenchmark(
    benchmark::State &state, ArrayRef<MemcmpOrBcmpConfiguration> configs) {
  const MemorySizeDistribution dist =
      ComparisonSetup::get_distributions()[state.range(0)];
  const MemcmpOrBcmpConfiguration config = configs[state.range(1)];
  std::discrete_distribution<size_t> size_sampler(dist.probabilities.begin(),
                                                  dist.probabilities.end());
  runComparisonBatch<true>(state, config, dist.probabilities.size() - 1,
                           size_sampler,
                           config.name.str() + "," + dist.name.str());
}

template <bool IsLatency>
void runSizeClassBenchmark(benchmark::State &state,
                           ArrayRef<MemcmpOrBcmpConfiguration> configs) {
  const size_t min_size = static_cast<size_t>(state.range(0));
  const size_t max_size = static_cast<size_t>(state.range(1));
  const MemcmpOrBcmpConfiguration config = configs[state.range(2)];
  std::uniform_int_distribution<size_t> size_sampler(min_size, max_size);
  runComparisonBatch<IsLatency>(state, config, max_size, size_sampler,
                                config.name.str() + ",[" +
                                    std::to_string(min_size) + "," +
                                    std::to_string(max_size) + "]");
}

void applyComparisonDistributionArgs(benchmark::Benchmark *b,
                                     int64_t conf_size) {
  const int64_t dist_size =
      static_cast<int64_t>(ComparisonSetup::get_distributions().size());
  for (int64_t dist_index = 0; dist_index < dist_size; ++dist_index)
    for (int64_t conf_index = 0; conf_index < conf_size; ++conf_index)
      b->Args({dist_index, conf_index});
}

void applySizeClassArgs(benchmark::Benchmark *b, int64_t conf_size) {
  static constexpr int64_t kSizeClasses[][2] = {
      {1, 3},   {4, 8},  {9, 16},   {17, 32},
      {33, 64}, {1, 16}, {65, 256}, {257, 1024},
  };
  b->ArgNames({"min", "max", "conf"});
  for (const auto &range : kSizeClasses)
    for (int64_t conf_index = 0; conf_index < conf_size; ++conf_index)
      b->Args({range[0], range[1], conf_index});
}

} // namespace

#define BENCHMARK_COMPARISON_SUITE(NAME, CONFIGS)                              \
  void BM_##NAME##Latency(benchmark::State &S) {                               \
    runDistributionLatencyBenchmark(S, CONFIGS);                               \
  }                                                                            \
  BENCHMARK(BM_##NAME##Latency)->Apply([](benchmark::Benchmark *b) {           \
    applyComparisonDistributionArgs(b, (CONFIGS).size());                      \
  });                                                                          \
  void BM_##NAME##SizeClassThroughput(benchmark::State &S) {                   \
    runSizeClassBenchmark<false>(S, CONFIGS);                                  \
  }                                                                            \
  BENCHMARK(BM_##NAME##SizeClassThroughput)                                    \
      ->Apply([](benchmark::Benchmark *b) {                                    \
        applySizeClassArgs(b, (CONFIGS).size());                               \
      });                                                                      \
  void BM_##NAME##SizeClassLatency(benchmark::State &S) {                      \
    runSizeClassBenchmark<true>(S, CONFIGS);                                   \
  }                                                                            \
  BENCHMARK(BM_##NAME##SizeClassLatency)->Apply([](benchmark::Benchmark *b) {  \
    applySizeClassArgs(b, (CONFIGS).size());                                   \
  })

BENCHMARK_COMPARISON_SUITE(Memcmp, getMemcmpConfigurations());
BENCHMARK_COMPARISON_SUITE(Bcmp, getBcmpConfigurations());
