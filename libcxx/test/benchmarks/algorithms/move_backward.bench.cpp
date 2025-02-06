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
void bm_ranges_move_backward_vb(benchmark::State& state) {
  auto n = state.range();
  std::vector<bool> v1(n, true);
  std::vector<bool> v2(n, false);
  benchmark::DoNotOptimize(v1);
  benchmark::DoNotOptimize(v2);
  std::vector<bool>* in  = &v1;
  std::vector<bool>* out = &v2;
  for (auto _ : state) {
    if constexpr (aligned) {
      benchmark::DoNotOptimize(std::ranges::move_backward(*in, std::ranges::end(*out)));
    } else {
      benchmark::DoNotOptimize(std::ranges::move_backward(*in | std::views::take(n - 4), std::ranges::end(*out)));
    }
    std::swap(in, out);
    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);
  }
}

template <bool aligned>
void bm_move_backward_vb(benchmark::State& state) {
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
    auto last2  = out->end();
    if constexpr (aligned) {
      benchmark::DoNotOptimize(std::move_backward(first1, last1, last2));
    } else {
      benchmark::DoNotOptimize(std::move_backward(first1, last1 - 4, last2));
    }
    std::swap(in, out);
    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);
  }
}

BENCHMARK(bm_ranges_move_backward_vb<true>)
    ->Name("bm_ranges_move_backward_vb_aligned")
    ->Range(8, 1 << 16)
    ->DenseRange(102400, 204800, 4096);
BENCHMARK(bm_ranges_move_backward_vb<false>)->Name("bm_ranges_move_backward_vb_unaligned")->Range(8, 1 << 20);

BENCHMARK(bm_move_backward_vb<true>)->Name("bm_move_backward_vb_aligned")->Range(8, 1 << 20);
BENCHMARK(bm_move_backward_vb<false>)->Name("bm_move_backward_vb_unaligned")->Range(8, 1 << 20);

BENCHMARK_MAIN();
