//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <benchmark/benchmark.h>
#include <vector>

static void bm_equal_iter(benchmark::State& state) {
  std::vector<char> vec1(state.range(), '1');
  std::vector<char> vec2(state.range(), '1');
  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::equal(vec1.begin(), vec1.end(), vec2.begin()));
  }
}
BENCHMARK(bm_equal_iter)->DenseRange(1, 8)->Range(16, 1 << 20);

static void bm_equal(benchmark::State& state) {
  std::vector<char> vec1(state.range(), '1');
  std::vector<char> vec2(state.range(), '1');
  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::equal(vec1.begin(), vec1.end(), vec2.begin(), vec2.end()));
  }
}
BENCHMARK(bm_equal)->DenseRange(1, 8)->Range(16, 1 << 20);

static void bm_ranges_equal(benchmark::State& state) {
  std::vector<char> vec1(state.range(), '1');
  std::vector<char> vec2(state.range(), '1');
  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::ranges::equal(vec1, vec2));
  }
}
BENCHMARK(bm_ranges_equal)->DenseRange(1, 8)->Range(16, 1 << 20);

static void bm_ranges_equal_vb_aligned(benchmark::State& state) {
  auto n = state.range();
  std::vector<bool> vec1(n, true);
  std::vector<bool> vec2(n, true);
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::ranges::equal(vec1, vec2));
    benchmark::DoNotOptimize(&vec1);
    benchmark::DoNotOptimize(&vec2);
  }
}

static void bm_ranges_equal_vb_unaligned(benchmark::State& state) {
  auto n = state.range();
  std::vector<bool> vec1(n, true);
  std::vector<bool> vec2(n + 8, true);
  auto beg1 = std::ranges::begin(vec1);
  auto end1 = std::ranges::end(vec1);
  auto beg2 = std::ranges::begin(vec2) + 4;
  auto end2 = std::ranges::end(vec2) - 4;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::ranges::equal(beg1, end1, beg2, end2));
    benchmark::DoNotOptimize(&vec1);
    benchmark::DoNotOptimize(&vec2);
  }
}

// Test std::ranges::equal for vector<bool>::iterator
BENCHMARK(bm_ranges_equal_vb_aligned)->RangeMultiplier(4)->Range(8, 1 << 20);
BENCHMARK(bm_ranges_equal_vb_unaligned)->Range(8, 1 << 20);

static void bm_equal_vb(benchmark::State& state, bool aligned) {
  auto n = state.range();
  std::vector<bool> vec1(n, true);
  std::vector<bool> vec2(aligned ? n : n + 8, true);
  auto beg1 = vec1.begin();
  auto end1 = vec1.end();
  auto beg2 = aligned ? vec2.begin() : vec2.begin() + 4;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::equal(beg1, end1, beg2));
    benchmark::DoNotOptimize(&vec1);
    benchmark::DoNotOptimize(&vec2);
  }
}

static void bm_equal_vb_aligned(benchmark::State& state) { bm_equal_vb(state, true); }
static void bm_equal_vb_unaligned(benchmark::State& state) { bm_equal_vb(state, false); }

// Test std::equal for vector<bool>::iterator
BENCHMARK(bm_equal_vb_aligned)->Range(8, 1 << 20);
BENCHMARK(bm_equal_vb_unaligned)->Range(8, 1 << 20);

BENCHMARK_MAIN();
