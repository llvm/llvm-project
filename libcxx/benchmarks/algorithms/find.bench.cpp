//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstring>
#include <random>
#include <vector>

template <class T>
static void bm_find(benchmark::State& state) {
  std::vector<T> vec1(state.range(), '1');
  std::mt19937_64 rng(std::random_device{}());

  for (auto _ : state) {
    auto idx  = rng() % vec1.size();
    vec1[idx] = '2';
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(std::find(vec1.begin(), vec1.end(), T('2')));
    vec1[idx] = '1';
  }
}
BENCHMARK(bm_find<char>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_find<short>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_find<int>)->DenseRange(1, 8)->Range(16, 1 << 20);

template <class T>
static void bm_ranges_find(benchmark::State& state) {
  std::vector<T> vec1(state.range(), '1');
  std::mt19937_64 rng(std::random_device{}());

  for (auto _ : state) {
    auto idx  = rng() % vec1.size();
    vec1[idx] = '2';
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(std::ranges::find(vec1, T('2')));
    vec1[idx] = '1';
  }
}
BENCHMARK(bm_ranges_find<char>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_ranges_find<short>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_ranges_find<int>)->DenseRange(1, 8)->Range(16, 1 << 20);

BENCHMARK_MAIN();
