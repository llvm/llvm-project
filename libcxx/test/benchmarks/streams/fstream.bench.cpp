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

static void bm_ofstream_write(benchmark::State& state) {
  std::vector<char> buffer;
  buffer.resize(16384);

  std::ofstream stream("/dev/null");

  for (auto _ : state)
    stream.write(buffer.data(), buffer.size());
}
BENCHMARK(bm_ofstream_write);

static void bm_ifstream_read(benchmark::State& state) {
  std::vector<char> buffer;
  buffer.resize(16384);

  std::ofstream gen_testfile("testfile");
  gen_testfile.write(buffer.data(), buffer.size());

  std::ifstream stream("testfile");
  assert(stream);

  for (auto _ : state) {
    stream.read(buffer.data(), buffer.size());
    benchmark::DoNotOptimize(buffer);
    stream.seekg(0);
  }
}
BENCHMARK(bm_ifstream_read);

BENCHMARK_MAIN();
