//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../GenerateInput.h"

template <class T>
static void bm_reverse(benchmark::State& state) {
  std::size_t const n = state.range();
  std::vector<T> vec;
  std::generate_n(std::back_inserter(vec), n, [] { return Generate<T>::cheap(); });
  for (auto _ : state) {
    std::reverse(vec.begin(), vec.end());
    benchmark::DoNotOptimize(vec);
  }
}
BENCHMARK(bm_reverse<int>)->Name("std::reverse(vector<int>)")->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_reverse<std::string>)->Name("std::reverse(vector<string>)")->DenseRange(1, 8)->Range(16, 1 << 20);

template <class T>
static void bm_ranges_reverse(benchmark::State& state) {
  std::size_t const n = state.range();
  std::vector<T> vec;
  std::generate_n(std::back_inserter(vec), n, [] { return Generate<T>::cheap(); });
  for (auto _ : state) {
    std::ranges::reverse(vec.begin(), vec.end());
    benchmark::DoNotOptimize(vec);
  }
}
BENCHMARK(bm_ranges_reverse<int>)->Name("ranges::reverse(vector<int>)")->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_ranges_reverse<std::string>)
    ->Name("ranges::reverse(vector<string>)")
    ->DenseRange(1, 8)
    ->Range(16, 1 << 20);

BENCHMARK_MAIN();
