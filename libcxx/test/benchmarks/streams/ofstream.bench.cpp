//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <vector>

#include <benchmark/benchmark.h>

static void bm_write(benchmark::State& state) {
  std::vector<char> buffer;
  buffer.resize(16384);

  std::ofstream stream("/dev/null");

  for (auto _ : state)
    stream.write(buffer.data(), buffer.size());
}
BENCHMARK(bm_write);

BENCHMARK_MAIN();
