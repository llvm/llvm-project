
#include "benchmark/benchmark.h"

#define BASIC_BENCHMARK_TEST(x) BENCHMARK(x)->Arg(8)->Arg(512)->Arg(8192)

void BM_empty(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(state.iterations());
  }
}
BENCHMARK(BM_empty);
BENCHMARK(BM_empty)->ThreadPerCpu();

void BM_spin_empty(benchmark::State& state) {
  for (auto _ : state) {
    for (int x = 0; x < state.range(0); ++x) {
      benchmark::DoNotOptimize(x);
    }
  }
}
BASIC_BENCHMARK_TEST(BM_spin_empty);
BASIC_BENCHMARK_TEST(BM_spin_empty)->ThreadPerCpu();

void BM_spin_pause_before(benchmark::State& state) {
  for (int i = 0; i < state.range(0); ++i) {
    benchmark::DoNotOptimize(i);
  }
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      benchmark::DoNotOptimize(i);
    }
  }
}
BASIC_BENCHMARK_TEST(BM_spin_pause_before);
BASIC_BENCHMARK_TEST(BM_spin_pause_before)->ThreadPerCpu();

void BM_spin_pause_during(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    for (int i = 0; i < state.range(0); ++i) {
      benchmark::DoNotOptimize(i);
    }
    state.ResumeTiming();
    for (int i = 0; i < state.range(0); ++i) {
      benchmark::DoNotOptimize(i);
    }
  }
}
BASIC_BENCHMARK_TEST(BM_spin_pause_during);
BASIC_BENCHMARK_TEST(BM_spin_pause_during)->ThreadPerCpu();

void BM_pause_during(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    state.ResumeTiming();
  }
}
BENCHMARK(BM_pause_during);
BENCHMARK(BM_pause_during)->ThreadPerCpu();
BENCHMARK(BM_pause_during)->UseRealTime();
BENCHMARK(BM_pause_during)->UseRealTime()->ThreadPerCpu();

void BM_spin_pause_after(benchmark::State& state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      benchmark::DoNotOptimize(i);
    }
  }
  for (int i = 0; i < state.range(0); ++i) {
    benchmark::DoNotOptimize(i);
  }
}
BASIC_BENCHMARK_TEST(BM_spin_pause_after);
BASIC_BENCHMARK_TEST(BM_spin_pause_after)->ThreadPerCpu();

void BM_spin_pause_before_and_after(benchmark::State& state) {
  for (int i = 0; i < state.range(0); ++i) {
    benchmark::DoNotOptimize(i);
  }
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      benchmark::DoNotOptimize(i);
    }
  }
  for (int i = 0; i < state.range(0); ++i) {
    benchmark::DoNotOptimize(i);
  }
}
BASIC_BENCHMARK_TEST(BM_spin_pause_before_and_after);
BASIC_BENCHMARK_TEST(BM_spin_pause_before_and_after)->ThreadPerCpu();

void BM_empty_stop_start(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_empty_stop_start);
BENCHMARK(BM_empty_stop_start)->ThreadPerCpu();


void BM_KeepRunning(benchmark::State& state) {
  size_t iter_count = 0;
  while (state.KeepRunning()) {
    ++iter_count;
  }
  assert(iter_count == state.max_iterations);
}
BENCHMARK(BM_KeepRunning);

void BM_RangedFor(benchmark::State& state) {
  size_t iter_count = 0;
  for (auto _ : state) {
    ++iter_count;
  }
  assert(iter_count == state.max_iterations);
}
BENCHMARK(BM_RangedFor);

BENCHMARK_MAIN();
