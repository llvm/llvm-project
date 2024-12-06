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

BENCHMARK(BM_Visit<3, 1>);
BENCHMARK(BM_Visit<3, 2>);
BENCHMARK(BM_Visit<3, 3>);
BENCHMARK(BM_Visit<3, 4>);
BENCHMARK(BM_Visit<3, 5>);
BENCHMARK(BM_Visit<3, 6>);
BENCHMARK(BM_Visit<3, 7>);
BENCHMARK(BM_Visit<3, 8>);
BENCHMARK(BM_Visit<3, 9>);
BENCHMARK(BM_Visit<3, 10>);
BENCHMARK(BM_Visit<3, 15>);
BENCHMARK(BM_Visit<3, 20>);

BENCHMARK_MAIN();
