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

BENCHMARK_TEMPLATE(BM_Visit, 3, 1);
BENCHMARK_TEMPLATE(BM_Visit, 3, 2);
BENCHMARK_TEMPLATE(BM_Visit, 3, 3);
BENCHMARK_TEMPLATE(BM_Visit, 3, 4);
BENCHMARK_TEMPLATE(BM_Visit, 3, 5);
BENCHMARK_TEMPLATE(BM_Visit, 3, 6);
BENCHMARK_TEMPLATE(BM_Visit, 3, 7);
BENCHMARK_TEMPLATE(BM_Visit, 3, 8);
BENCHMARK_TEMPLATE(BM_Visit, 3, 9);
BENCHMARK_TEMPLATE(BM_Visit, 3, 10);
BENCHMARK_TEMPLATE(BM_Visit, 3, 15);
BENCHMARK_TEMPLATE(BM_Visit, 3, 20);

BENCHMARK_MAIN();
