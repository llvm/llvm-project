#include "benchmark/benchmark_api.h"

#include <chrono>
#include <thread>

void BM_basic(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
}

void BM_basic_slow(benchmark::State& state) {
  std::chrono::milliseconds sleep_duration(state.range_x());
  while (state.KeepRunning()) {
    std::this_thread::sleep_for(
      std::chrono::duration_cast<std::chrono::nanoseconds>(sleep_duration)
      );
  }
}

BENCHMARK(BM_basic);
BENCHMARK(BM_basic)->Arg(42);
BENCHMARK(BM_basic_slow)->Arg(10)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_basic_slow)->Arg(100)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_basic_slow)->Arg(1000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_basic)->Range(1, 8);
BENCHMARK(BM_basic)->RangeMultiplier(2)->Range(1, 8);
BENCHMARK(BM_basic)->DenseRange(10, 15);
BENCHMARK(BM_basic)->ArgPair(42, 42);
BENCHMARK(BM_basic)->RangePair(64, 512, 64, 512);
BENCHMARK(BM_basic)->MinTime(0.7);
BENCHMARK(BM_basic)->UseRealTime();
BENCHMARK(BM_basic)->ThreadRange(2, 4);
BENCHMARK(BM_basic)->ThreadPerCpu();
BENCHMARK(BM_basic)->Repetitions(3);

void CustomArgs(benchmark::internal::Benchmark* b) {
  for (int i = 0; i < 10; ++i) {
    b->Arg(i);
  }
}

BENCHMARK(BM_basic)->Apply(CustomArgs);

BENCHMARK_MAIN()
