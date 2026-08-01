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

BENCHMARK(BM_adjacent_full<2>)->Name("rng::adjacent_view::begin()/2 (full view)");
BENCHMARK(BM_adjacent_full<10>)->Name("rng::adjacent_view::begin()/10 (full view)");
BENCHMARK(BM_adjacent_full<100>)->Name("rng::adjacent_view::begin()/100 (full view)");
BENCHMARK(BM_adjacent_full<1000>)->Name("rng::adjacent_view::begin()/1000 (full view)");

template <size_t N>
void BM_adjacent_empty(benchmark::State& state) {
  const std::vector<int> inputs;
  auto view = inputs | std::views::adjacent<N>;
  for (auto _ : state) {
    auto it = view.begin();
    benchmark::DoNotOptimize(it);
  }
}

BENCHMARK(BM_adjacent_empty<1000>)->Name("rng::adjacent_view::begin()/1000 (empty view)");

} // namespace

BENCHMARK_MAIN();
