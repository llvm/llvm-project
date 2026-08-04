
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <ios>
#include <locale>

#include <benchmark/benchmark.h>

static void BM_num_get(benchmark::State& state) {
  std::locale loc1, loc2;

  for (auto _ : state) {
    benchmark::DoNotOptimize(loc1);
    benchmark::DoNotOptimize(loc2);
    std::swap(loc1, loc2);
  }
}
BENCHMARK(BM_num_get)->Name("std::swap(std::locale&, std::locale&)");

BENCHMARK_MAIN();
