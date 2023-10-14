//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <exception>
#include <stdexcept>

void bm_make_exception_ptr(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::make_exception_ptr(std::runtime_error{"Some error"}));
  }
}
BENCHMARK(bm_make_exception_ptr);

BENCHMARK_MAIN();
