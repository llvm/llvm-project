//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <memory>

#include "benchmark/benchmark.h"
#include "test_macros.h"

static void BM_align(benchmark::State& state) {
  char buffer[1024];
  char* data = buffer + 123;
  std::size_t sz{1024 - 123};
  void* p;

  for (auto _ : state) {
    benchmark::DoNotOptimize(std::align(state.range(), state.range(), p, sz))
  }
}
BENCHMARK(BM_align)->Range(1, 256);

BENCHMARK_MAIN();
