//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <iterator>
#include <vector>

#include "test_iterators.h"

static void bm_shift_left_random_access_range_with_sized_sentinel(benchmark::State& state) {
  std::vector<int> a(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);

    auto begin = random_access_iterator(a.data());
    auto end   = random_access_iterator(a.data() + a.size());

    static_assert(std::random_access_iterator<decltype(begin)>);
    static_assert(std::sized_sentinel_for<decltype(end), decltype(begin)>);

    benchmark::DoNotOptimize(std::ranges::shift_left(begin, end, a.size() / 2));
  }
}
BENCHMARK(bm_shift_left_random_access_range_with_sized_sentinel)->RangeMultiplier(16)->Range(16, 16 << 20);

static void bm_shift_left_forward_range_with_sized_sentinel(benchmark::State& state) {
  std::vector<int> a(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);

    auto begin = forward_iterator(a.data());
    auto end   = sized_sentinel(forward_iterator(a.data() + a.size()));

    static_assert(!std::random_access_iterator<decltype(begin)>);
    static_assert(std::sized_sentinel_for<decltype(end), decltype(begin)>);

    benchmark::DoNotOptimize(std::ranges::shift_left(begin, end, a.size() / 2));
  }
}
BENCHMARK(bm_shift_left_forward_range_with_sized_sentinel)->RangeMultiplier(16)->Range(16, 16 << 20);

static void bm_shift_left_forward_range(benchmark::State& state) {
  std::vector<int> a(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);

    auto begin = forward_iterator(a.data());
    auto end   = forward_iterator(a.data() + a.size());

    static_assert(!std::random_access_iterator<decltype(begin)>);
    static_assert(!std::sized_sentinel_for<decltype(end), decltype(begin)>);

    benchmark::DoNotOptimize(std::ranges::shift_left(begin, end, a.size() / 2));
  }
}
BENCHMARK(bm_shift_left_forward_range)->RangeMultiplier(16)->Range(16, 16 << 20);

BENCHMARK_MAIN();
