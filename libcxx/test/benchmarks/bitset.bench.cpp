//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <bitset>

static void BM_ctor_ull(benchmark::State& state) {
  unsigned long long val = (1ULL << state.range(0)) - 1;
  for (auto _ : state) {
    std::bitset<128> b(val);
    benchmark::DoNotOptimize(b);
  }
}

BENCHMARK(BM_ctor_ull)->DenseRange(1, 63);

BENCHMARK_MAIN();
