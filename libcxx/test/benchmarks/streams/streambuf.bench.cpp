//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <fstream>
#include <vector>

#include <benchmark/benchmark.h>
#include "test_macros.h"

namespace {
struct testbuf : std::streambuf {};
} // namespace

static TEST_ALIGN_BENCHMARK void bm_pubimbue(benchmark::State& state) {
  testbuf buf;

  auto loc = std::locale::classic();

  for (auto _ : state) {
    benchmark::DoNotOptimize(buf);
    benchmark::DoNotOptimize(loc);
    buf.pubimbue(loc);
  }
}
BENCHMARK(bm_pubimbue)->Name("std::streambuf::pubimbue(const std::locale&)");

BENCHMARK_MAIN();
