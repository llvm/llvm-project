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

static void bm_ranges_rotate_vb(benchmark::State& state) {
  auto n = state.range();
  std::vector<bool> v(n);
  auto mid = std::ranges::begin(v) + n / 2;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::ranges::rotate(v, mid));
    benchmark::DoNotOptimize(&v);
  }
}

static void bm_rotate_vb(benchmark::State& state) {
  auto n = state.range();
  std::vector<bool> v(n);
  auto beg = v.begin();
  auto mid = v.begin() + n / 2;
  auto end = v.end();
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::rotate(beg, mid, end));
    benchmark::DoNotOptimize(&v);
  }
}

BENCHMARK(bm_ranges_rotate_vb)->Range(8, 1 << 20);
BENCHMARK(bm_rotate_vb)->Range(8, 1 << 20);

BENCHMARK_MAIN();
