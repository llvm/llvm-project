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

BENCHMARK_TEMPLATE(BM_Visit, 2, 1);
BENCHMARK_TEMPLATE(BM_Visit, 2, 2);
BENCHMARK_TEMPLATE(BM_Visit, 2, 3);
BENCHMARK_TEMPLATE(BM_Visit, 2, 4);
BENCHMARK_TEMPLATE(BM_Visit, 2, 5);
BENCHMARK_TEMPLATE(BM_Visit, 2, 6);
BENCHMARK_TEMPLATE(BM_Visit, 2, 7);
BENCHMARK_TEMPLATE(BM_Visit, 2, 8);
BENCHMARK_TEMPLATE(BM_Visit, 2, 9);
BENCHMARK_TEMPLATE(BM_Visit, 2, 10);
BENCHMARK_TEMPLATE(BM_Visit, 2, 20);
BENCHMARK_TEMPLATE(BM_Visit, 2, 30);
BENCHMARK_TEMPLATE(BM_Visit, 2, 40);
BENCHMARK_TEMPLATE(BM_Visit, 2, 50);

BENCHMARK_MAIN();
