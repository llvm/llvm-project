//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <deque>
#include <ranges>
#include <vector>

#include "benchmark/benchmark.h"

namespace {

template <size_t N>
void BM_adjacent_full(benchmark::State& state) {
  const std::vector<int> inputs(1000000, 42);
  auto view = inputs | std::views::adjacent<N>;
  for (auto _ : state) {
    auto it = view.begin();
    benchmark::DoNotOptimize(it);
  }
}

BENCHMARK(BM_adjacent_full<2>);
BENCHMARK(BM_adjacent_full<3>);
BENCHMARK(BM_adjacent_full<4>);
BENCHMARK(BM_adjacent_full<5>);
BENCHMARK(BM_adjacent_full<6>);
BENCHMARK(BM_adjacent_full<7>);
BENCHMARK(BM_adjacent_full<8>);
BENCHMARK(BM_adjacent_full<9>);
BENCHMARK(BM_adjacent_full<10>);
BENCHMARK(BM_adjacent_full<100>);
BENCHMARK(BM_adjacent_full<1000>);

template <size_t N>
void BM_adjacent_empty(benchmark::State& state) {
  const std::vector<int> inputs;
  auto view = inputs | std::views::adjacent<N>;
  for (auto _ : state) {
    auto it = view.begin();
    benchmark::DoNotOptimize(it);
  }
}

BENCHMARK(BM_adjacent_empty<2>);
BENCHMARK(BM_adjacent_empty<3>);
BENCHMARK(BM_adjacent_empty<4>);
BENCHMARK(BM_adjacent_empty<5>);
BENCHMARK(BM_adjacent_empty<6>);
BENCHMARK(BM_adjacent_empty<7>);
BENCHMARK(BM_adjacent_empty<8>);
BENCHMARK(BM_adjacent_empty<9>);
BENCHMARK(BM_adjacent_empty<10>);
BENCHMARK(BM_adjacent_empty<100>);
BENCHMARK(BM_adjacent_empty<1000>);

} // namespace

BENCHMARK_MAIN();