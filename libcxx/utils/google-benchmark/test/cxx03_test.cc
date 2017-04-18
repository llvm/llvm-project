#undef NDEBUG
#include <cassert>
#include <cstddef>

#include "benchmark/benchmark.h"

#if __cplusplus >= 201103L
#error C++11 or greater detected. Should be C++03.
#endif

void BM_empty(benchmark::State& state) {
  while (state.KeepRunning()) {
    volatile std::size_t x = state.iterations();
    ((void)x);
  }
}
BENCHMARK(BM_empty);

// The new C++11 interface for args/ranges requires initializer list support.
// Therefore we provide the old interface to support C++03.
void BM_old_arg_range_interface(benchmark::State& state) {
  assert((state.range(0) == 1 && state.range(1) == 2) ||
         (state.range(0) == 5 && state.range(1) == 6));
  while (state.KeepRunning()) {
  }
}
BENCHMARK(BM_old_arg_range_interface)->ArgPair(1, 2)->RangePair(5, 5, 6, 6);

template <class T, class U>
void BM_template2(benchmark::State& state) {
  BM_empty(state);
}
BENCHMARK_TEMPLATE2(BM_template2, int, long);

template <class T>
void BM_template1(benchmark::State& state) {
  BM_empty(state);
}
BENCHMARK_TEMPLATE(BM_template1, long);
BENCHMARK_TEMPLATE1(BM_template1, int);

void BM_counters(benchmark::State& state) {
    BM_empty(state);
    state.counters["Foo"] = 2;
}
BENCHMARK(BM_counters);

BENCHMARK_MAIN()
