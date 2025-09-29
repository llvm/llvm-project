//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <vector>

static void BM_vector_bool_size_ctor(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<bool> vec(100, true);
    benchmark::DoNotOptimize(vec);
  }
}
BENCHMARK(BM_vector_bool_size_ctor);

BENCHMARK_MAIN();
