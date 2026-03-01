//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iterator>
#include <fstream>
#include <vector>

#include <benchmark/benchmark.h>

static void bm_copy(benchmark::State& state) {
  std::vector<char> buffer;
  buffer.resize(16384);

  std::ofstream stream("/dev/null");

  for (auto _ : state)
    std::copy(buffer.begin(), buffer.end(), std::ostreambuf_iterator<char>(stream.rdbuf()));
}
BENCHMARK(bm_copy)->Name("std::copy(CharT*, CharT*, ostreambuf_iterator)");

BENCHMARK_MAIN();
