#include <benchmark/benchmark.h>

template <class Iter, class ValueT>
Iter my_find(Iter first, Iter last, const ValueT& i) {
  for (; first != last; ++first) {
    if (*first == i)
      break;
  }
  return first;
}

static auto bm_find_if_no_vectorization(benchmark::State& state) {
  std::size_t const size = 8192;
  std::vector<short> c(size, 0);

  for ([[maybe_unused]] auto _ : state) {
    benchmark::DoNotOptimize(c);
    std::vector<short>::iterator result;
    result = my_find(c.begin(), c.end(), 1);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(bm_find_if_no_vectorization);

static auto bm_find_if_autovectorization(benchmark::State& state) {
  std::size_t const size = 8192;
  std::vector<short> c(size, 0);

  for ([[maybe_unused]] auto _ : state) {
    benchmark::DoNotOptimize(c);
    std::vector<short>::iterator result;
    result = find_if(c.begin(), c.end(), [](short i) { return i == 1; });
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(bm_find_if_autovectorization);

static auto bm_find_manual_vectorization(benchmark::State& state) {
  std::size_t const size = 8192;
  std::vector<short> c(size, 0);

  for ([[maybe_unused]] auto _ : state) {
    benchmark::DoNotOptimize(c);
    std::vector<short>::iterator result;
    result = find(c.begin(), c.end(), 1);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(bm_find_manual_vectorization);

BENCHMARK_MAIN();
