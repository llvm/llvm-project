//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <bitset>

#include "benchmark/benchmark.h"

template <std::size_t N>
static void bm_left_shift(benchmark::State& state) {
  std::bitset<N> b;

  for (auto _ : state) {
    b <<= 4;
    benchmark::DoNotOptimize(b);
  }
}
BENCHMARK(bm_left_shift<32>);
BENCHMARK(bm_left_shift<64>);

template <std::size_t N>
static void bm_right_shift(benchmark::State& state) {
  std::bitset<N> b;

  for (auto _ : state) {
    b >>= 4;
    benchmark::DoNotOptimize(b);
  }
}
BENCHMARK(bm_right_shift<32>);
BENCHMARK(bm_right_shift<64>);

BENCHMARK_MAIN();
