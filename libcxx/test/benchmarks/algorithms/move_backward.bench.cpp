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
#include <vector>

static void bm_ranges_move_backward_vb(benchmark::State& state, bool aligned) {
  auto n = state.range();
  std::vector<bool> in(n, true);
  std::vector<bool> out(aligned ? n : n + 8);
  benchmark::DoNotOptimize(&in);
  auto dst = aligned ? out.end() : out.end() - 4;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::ranges::move_backward(in, dst));
    benchmark::DoNotOptimize(&out);
  }
}

static void bm_move_backward(benchmark::State& state, bool aligned) {
  auto n = state.range();
  std::vector<bool> in(n, true);
  std::vector<bool> out(aligned ? n : n + 8);
  benchmark::DoNotOptimize(&in);
  auto beg = in.begin();
  auto end = in.end();
  auto dst = aligned ? out.end() : out.end() - 4;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::move_backward(beg, end, dst));
    benchmark::DoNotOptimize(&out);
  }
}

static void bm_ranges_move_backward_vb_aligned(benchmark::State& state) { bm_ranges_move_backward_vb(state, true); }
static void bm_ranges_move_backward_vb_unaligned(benchmark::State& state) { bm_ranges_move_backward_vb(state, false); }

static void bm_move_backward_vb_aligned(benchmark::State& state) { bm_move_backward(state, true); }
static void bm_move_backward_vb_unaligned(benchmark::State& state) { bm_move_backward(state, false); }

// Test std::ranges::move_backward for vector<bool>::iterator
BENCHMARK(bm_ranges_move_backward_vb_aligned)->Range(8, 1 << 16)->DenseRange(102400, 204800, 4096);
BENCHMARK(bm_ranges_move_backward_vb_unaligned)->Range(8, 1 << 20);

// Test std::move_backward for vector<bool>::iterator
BENCHMARK(bm_move_backward_vb_aligned)->Range(8, 1 << 20);
BENCHMARK(bm_move_backward_vb_unaligned)->Range(8, 1 << 20);

BENCHMARK_MAIN();
