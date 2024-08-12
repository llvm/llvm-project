//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <vector>

// Benchmarks the worst case: check the whole range just to find out that they compare equal
template <class T>
static void bm_lexicographical_compare(benchmark::State& state) {
  std::vector<T> vec1(state.range(), '1');
  std::vector<T> vec2(state.range(), '1');

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::lexicographical_compare(vec1.begin(), vec1.end(), vec2.begin(), vec2.end()));
  }
}
BENCHMARK(bm_lexicographical_compare<unsigned char>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_lexicographical_compare<signed char>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_lexicographical_compare<int>)->DenseRange(1, 8)->Range(16, 1 << 20);

template <class T>
static void bm_ranges_lexicographical_compare(benchmark::State& state) {
  std::vector<T> vec1(state.range(), '1');
  std::vector<T> vec2(state.range(), '1');

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::ranges::lexicographical_compare(vec1.begin(), vec1.end(), vec2.begin(), vec2.end()));
  }
}
BENCHMARK(bm_ranges_lexicographical_compare<unsigned char>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_ranges_lexicographical_compare<signed char>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_ranges_lexicographical_compare<int>)->DenseRange(1, 8)->Range(16, 1 << 20);

BENCHMARK_MAIN();
