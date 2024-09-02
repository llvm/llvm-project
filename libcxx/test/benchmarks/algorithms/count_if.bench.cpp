//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <deque>

static void bm_deque_count_if(benchmark::State& state) {
  std::deque<char> deque1(state.range(), '1');
  for (auto _ : state) {
    benchmark::DoNotOptimize(deque1);
    benchmark::DoNotOptimize(std::count_if(deque1.begin(), deque1.end(), [](char& v) { return v == '0'; }));
  }
}
BENCHMARK(bm_deque_count_if)->DenseRange(1, 8)->Range(16, 1 << 20);

BENCHMARK_MAIN();
