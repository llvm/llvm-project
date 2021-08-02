#include "LibcBenchmark.h"
#include "LibcMemoryBenchmark.h"
#include "MemorySizeDistributions.h"
#include "benchmark/benchmark.h"
#include <cstdint>
#include <random>
#include <vector>

namespace __llvm_libc {

extern void *memcpy(void *__restrict, const void *__restrict, size_t);
extern void *memset(void *, int, size_t);
extern void bzero(void *, size_t);
extern int memcmp(const void *, const void *, size_t);

} // namespace __llvm_libc

using llvm::Align;
using llvm::ArrayRef;
using llvm::libc_benchmarks::ComparisonHarness;
using llvm::libc_benchmarks::CopyHarness;
using llvm::libc_benchmarks::MemorySizeDistribution;
using llvm::libc_benchmarks::OffsetDistribution;
using llvm::libc_benchmarks::SetHarness;

static constexpr Align kBenchmarkAlignment = Align::Constant<1>();

template <typename Harness> struct Randomized : public Harness {
  Randomized(benchmark::State &State)
      : State(State), Distribution(Harness::getDistributions()[State.range(0)]),
        Probabilities(Distribution.Probabilities),
        SizeSampler(Probabilities.begin(), Probabilities.end()),
        OffsetSampler(Harness::BufferSize, Probabilities.size() - 1,
                      kBenchmarkAlignment) {
    for (auto &P : Harness::Parameters) {
      P.OffsetBytes = OffsetSampler(Gen);
      P.SizeBytes = SizeSampler(Gen);
      Harness::checkValid(P);
    }
  }

  ~Randomized() {
    const size_t AvgBytesPerIteration =
        Harness::getBatchBytes() / Harness::BatchSize;
    const size_t TotalBytes = State.iterations() * AvgBytesPerIteration;
    State.SetBytesProcessed(TotalBytes);
    State.SetLabel(Distribution.Name.str());
    State.counters["bytes_per_cycle"] = benchmark::Counter(
        TotalBytes / benchmark::CPUInfo::Get().cycles_per_second,
        benchmark::Counter::kIsRate);
  }

  template <typename Function> inline void runBatch(Function foo) {
    for (const auto &P : Harness::Parameters)
      benchmark::DoNotOptimize(Harness::Call(P, foo));
  }

private:
  benchmark::State &State;
  Harness UP;
  MemorySizeDistribution Distribution;
  ArrayRef<double> Probabilities;
  std::discrete_distribution<unsigned> SizeSampler;
  OffsetDistribution OffsetSampler;
  std::mt19937_64 Gen;
};

template <typename Harness> static int64_t getMaxIndex() {
  return Harness::getDistributions().size() - 1;
}

void BM_Memcpy(benchmark::State &State) {
  Randomized<CopyHarness> Harness(State);
  while (State.KeepRunningBatch(Harness.BatchSize))
    Harness.runBatch(__llvm_libc::memcpy);
}
BENCHMARK(BM_Memcpy)->DenseRange(0, getMaxIndex<CopyHarness>());

void BM_Memcmp(benchmark::State &State) {
  Randomized<ComparisonHarness> Harness(State);
  while (State.KeepRunningBatch(Harness.BatchSize))
    Harness.runBatch(__llvm_libc::memcmp);
}
BENCHMARK(BM_Memcmp)->DenseRange(0, getMaxIndex<ComparisonHarness>());

void BM_Memset(benchmark::State &State) {
  Randomized<SetHarness> Harness(State);
  while (State.KeepRunningBatch(Harness.BatchSize))
    Harness.runBatch(__llvm_libc::memset);
}
BENCHMARK(BM_Memset)->DenseRange(0, getMaxIndex<SetHarness>());

void BM_Bzero(benchmark::State &State) {
  Randomized<SetHarness> Harness(State);
  while (State.KeepRunningBatch(Harness.BatchSize))
    Harness.runBatch(__llvm_libc::bzero);
}
BENCHMARK(BM_Bzero)->DenseRange(0, getMaxIndex<SetHarness>());
