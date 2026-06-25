#include <algorithm>
#include <limits>
#include <vector>

#include <benchmark/benchmark.h>
#include <random>

template <typename T>
static void BM_stdmin_element_decreasing(benchmark::State& state) {
  std::vector<T> v(state.range(0));
  T start = std::numeric_limits<T>::max();
  T end   = std::numeric_limits<T>::min();

  for (size_t i = 0; i < v.size(); i++)
    v[i] = ((start != end) ? start-- : end);

  for (auto _ : state) {
    benchmark::DoNotOptimize(v);
    benchmark::DoNotOptimize(std::min_element(v.begin(), v.end()));
  }
}

BENCHMARK(BM_stdmin_element_decreasing<char>)
    ->DenseRange(1, 8)
    ->Range(32, 128)
    ->Range(256, 4096)
    ->DenseRange(5000, 10000, 1000)
    ->Range(1 << 14, 1 << 16)
    ->Arg(70000);
BENCHMARK(BM_stdmin_element_decreasing<short>)
    ->DenseRange(1, 8)
    ->Range(32, 128)
    ->Range(256, 4096)
    ->DenseRange(5000, 10000, 1000)
    ->Range(1 << 14, 1 << 16)
    ->Arg(70000);
BENCHMARK(BM_stdmin_element_decreasing<int>)
    ->DenseRange(1, 8)
    ->Range(32, 128)
    ->Range(256, 4096)
    ->DenseRange(5000, 10000, 1000)
    ->Range(1 << 14, 1 << 16)
    ->Arg(70000);
BENCHMARK(BM_stdmin_element_decreasing<long long>)
    ->DenseRange(1, 8)
    ->Range(32, 128)
    ->Range(256, 4096)
    ->DenseRange(5000, 10000, 1000)
    ->Range(1 << 14, 1 << 16)
    ->Arg(70000);
BENCHMARK(BM_stdmin_element_decreasing<unsigned char>)
    ->DenseRange(1, 8)
    ->Range(32, 128)
    ->Range(256, 4096)
    ->DenseRange(5000, 10000, 1000)
    ->Range(1 << 14, 1 << 16)
    ->Arg(70000);
BENCHMARK(BM_stdmin_element_decreasing<unsigned short>)
    ->DenseRange(1, 8)
    ->Range(32, 128)
    ->Range(256, 4096)
    ->DenseRange(5000, 10000, 1000)
    ->Range(1 << 14, 1 << 16)
    ->Arg(70000);
BENCHMARK(BM_stdmin_element_decreasing<unsigned int>)
    ->DenseRange(1, 8)
    ->Range(32, 128)
    ->Range(256, 4096)
    ->DenseRange(5000, 10000, 1000)
    ->Range(1 << 14, 1 << 16)
    ->Arg(70000);
BENCHMARK(BM_stdmin_element_decreasing<unsigned long long>)
    ->DenseRange(1, 8)
    ->Range(32, 128)
    ->Range(256, 4096)
    ->DenseRange(5000, 10000, 1000)
    ->Range(1 << 14, 1 << 16)
    ->Arg(70000);

BENCHMARK_MAIN();
