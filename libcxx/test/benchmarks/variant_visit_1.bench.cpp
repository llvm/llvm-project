//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"

#include "VariantBenchmarks.h"

using namespace VariantBenchmarks;

BENCHMARK(BM_Visit<1, 1>);
BENCHMARK(BM_Visit<1, 2>);
BENCHMARK(BM_Visit<1, 3>);
BENCHMARK(BM_Visit<1, 4>);
BENCHMARK(BM_Visit<1, 5>);
BENCHMARK(BM_Visit<1, 6>);
BENCHMARK(BM_Visit<1, 7>);
BENCHMARK(BM_Visit<1, 8>);
BENCHMARK(BM_Visit<1, 9>);
BENCHMARK(BM_Visit<1, 10>);
BENCHMARK(BM_Visit<1, 20>);
BENCHMARK(BM_Visit<1, 30>);
BENCHMARK(BM_Visit<1, 40>);
BENCHMARK(BM_Visit<1, 50>);
BENCHMARK(BM_Visit<1, 60>);
BENCHMARK(BM_Visit<1, 70>);
BENCHMARK(BM_Visit<1, 80>);
BENCHMARK(BM_Visit<1, 90>);
BENCHMARK(BM_Visit<1, 100>);

BENCHMARK_MAIN();
