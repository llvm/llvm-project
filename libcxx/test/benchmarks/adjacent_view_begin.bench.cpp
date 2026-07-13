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
BENCHMARK(BM_adjacent_full<3>)->Name("rng::adjacent_view::begin()/3 (full view)");
BENCHMARK(BM_adjacent_full<4>)->Name("rng::adjacent_view::begin()/4 (full view)");
BENCHMARK(BM_adjacent_full<5>)->Name("rng::adjacent_view::begin()/5 (full view)");
BENCHMARK(BM_adjacent_full<6>)->Name("rng::adjacent_view::begin()/6 (full view)");
BENCHMARK(BM_adjacent_full<7>)->Name("rng::adjacent_view::begin()/7 (full view)");
BENCHMARK(BM_adjacent_full<8>)->Name("rng::adjacent_view::begin()/8 (full view)");
BENCHMARK(BM_adjacent_full<9>)->Name("rng::adjacent_view::begin()/9 (full view)");
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

BENCHMARK(BM_adjacent_empty<2>)->Name("rng::adjacent_view::begin()/2 (empty view)");
BENCHMARK(BM_adjacent_empty<3>)->Name("rng::adjacent_view::begin()/3 (empty view)");
BENCHMARK(BM_adjacent_empty<4>)->Name("rng::adjacent_view::begin()/4 (empty view)");
BENCHMARK(BM_adjacent_empty<5>)->Name("rng::adjacent_view::begin()/5 (empty view)");
BENCHMARK(BM_adjacent_empty<6>)->Name("rng::adjacent_view::begin()/6 (empty view)");
BENCHMARK(BM_adjacent_empty<7>)->Name("rng::adjacent_view::begin()/7 (empty view)");
BENCHMARK(BM_adjacent_empty<8>)->Name("rng::adjacent_view::begin()/8 (empty view)");
BENCHMARK(BM_adjacent_empty<9>)->Name("rng::adjacent_view::begin()/9 (empty view)");
BENCHMARK(BM_adjacent_empty<10>)->Name("rng::adjacent_view::begin()/10 (empty view)");
BENCHMARK(BM_adjacent_empty<100>)->Name("rng::adjacent_view::begin()/100 (empty view)");
BENCHMARK(BM_adjacent_empty<1000>)->Name("rng::adjacent_view::begin()/1000 (empty view)");

} // namespace

BENCHMARK_MAIN();
