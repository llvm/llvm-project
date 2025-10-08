//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <benchmark/benchmark.h>
#include <exception>

void bm_make_exception_ptr(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::make_exception_ptr(42));
  }
}
BENCHMARK(bm_make_exception_ptr)->ThreadRange(1, 8);


void bm_empty_exception_ptr(benchmark::State& state) {
  for (auto _ : state) {
    // All of the following operations are no-ops because
    // the exception_ptr is empty. Hence, the compiler should
    // be able to optimize them very aggressively.
    std::exception_ptr p1;
    std::exception_ptr p2 (p1);
    std::exception_ptr p3 (std::move(p2));
    p2 = std::move(p1);
    p1 = p2;
    swap(p1, p2);
    benchmark::DoNotOptimize(p1 == nullptr && nullptr == p2 && p1 == p2);
  }
}
BENCHMARK(bm_empty_exception_ptr);

BENCHMARK_MAIN();
