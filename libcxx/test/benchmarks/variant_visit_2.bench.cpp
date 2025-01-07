//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include "benchmark/benchmark.h"

#include "VariantBenchmarks.h"

using namespace VariantBenchmarks;

BENCHMARK(BM_Visit<2, 1>);
BENCHMARK(BM_Visit<2, 2>);
BENCHMARK(BM_Visit<2, 3>);
BENCHMARK(BM_Visit<2, 4>);
BENCHMARK(BM_Visit<2, 5>);
BENCHMARK(BM_Visit<2, 6>);
BENCHMARK(BM_Visit<2, 7>);
BENCHMARK(BM_Visit<2, 8>);
BENCHMARK(BM_Visit<2, 9>);
BENCHMARK(BM_Visit<2, 10>);
BENCHMARK(BM_Visit<2, 20>);
BENCHMARK(BM_Visit<2, 30>);
BENCHMARK(BM_Visit<2, 40>);
BENCHMARK(BM_Visit<2, 50>);

BENCHMARK_MAIN();
