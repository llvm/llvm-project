//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <benchmark/benchmark.h>
#include <ranges>
#include <vector>

template <bool aligned>
void bm_ranges_move_vb(benchmark::State& state) {
  auto n = state.range();
  std::vector<bool> v1(n, true);
  std::vector<bool> v2(n, false);
  benchmark::DoNotOptimize(v1);
  benchmark::DoNotOptimize(v2);
  std::vector<bool>* in  = &v1;
  std::vector<bool>* out = &v2;
  for (auto _ : state) {
    if constexpr (aligned) {
      benchmark::DoNotOptimize(std::ranges::move(*in, std::ranges::begin(*out)));
    } else {
      benchmark::DoNotOptimize(std::ranges::move(*in | std::views::drop(4), std::ranges::begin(*out)));
    }
    std::swap(in, out);
    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);
  }
}

template <bool aligned>
void bm_move_vb(benchmark::State& state) {
  auto n = state.range();
  std::vector<bool> v1(n, true);
  std::vector<bool> v2(n, false);
  benchmark::DoNotOptimize(v1);
  benchmark::DoNotOptimize(v2);
  std::vector<bool>* in  = &v1;
  std::vector<bool>* out = &v2;
  for (auto _ : state) {
    auto first1 = in->begin();
    auto last1  = in->end();
    auto first2 = out->begin();
    if constexpr (aligned) {
      benchmark::DoNotOptimize(std::move(first1, last1, first2));
    } else {
      benchmark::DoNotOptimize(std::move(first1 + 4, last1, first2));
    }
    std::swap(in, out);
    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);
  }
}

BENCHMARK(bm_ranges_move_vb<true>)
    ->Name("bm_ranges_move_vb_aligned")
    ->Range(8, 1 << 16)
    ->DenseRange(102400, 204800, 4096);
BENCHMARK(bm_ranges_move_vb<false>)->Name("bm_ranges_move_vb_unaligned")->Range(8, 1 << 20);

BENCHMARK(bm_move_vb<true>)->Name("bm_move_vb_aligned")->Range(8, 1 << 20);
BENCHMARK(bm_move_vb<false>)->Name("bm_move_vb_unaligned")->Range(8, 1 << 20);

BENCHMARK_MAIN();
