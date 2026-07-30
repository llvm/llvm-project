//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <cassert>
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
BENCHMARK(bm_ofstream_write)->Name("std::ofstream::write(char*, size)");

static void bm_ifstream_read(benchmark::State& state) {
  std::vector<char> buffer;
  buffer.resize(16384);

  {
    std::ofstream gen_testfile("testfile");
    gen_testfile.write(buffer.data(), buffer.size());
  }

  std::ifstream stream("testfile");
  assert(stream);

  for (auto _ : state) {
    stream.read(buffer.data(), buffer.size());
    benchmark::DoNotOptimize(buffer);
    stream.seekg(0);
  }
}
BENCHMARK(bm_ifstream_read)->Name("std::ifstream::read(char*, size)");

void run_sizes(benchmark::Benchmark* benchmark) { benchmark->Arg(0)->Arg(100)->Arg(4000)->Arg(10000); }

template <bool prime>
static void bm_seekoff_cur(benchmark::State& state) {
  std::vector<char> buffer;
  buffer.resize(16384);

  {
    std::ofstream gen_testfile("testfile");
    gen_testfile.write(buffer.data(), buffer.size());
  }

  std::ifstream stream("testfile");
  assert(stream);

  auto val = state.range();

  for (auto _ : state) {
    if constexpr (prime)
      benchmark::DoNotOptimize(stream.rdbuf()->sgetc());
    benchmark::DoNotOptimize(stream.seekg(val, std::ios::cur));
    val = -val;
  }
}
BENCHMARK(bm_seekoff_cur<true>)->Name("std::ifstream::seekg(N, std::ios::cur) (primed buffer)")->Apply(run_sizes);
BENCHMARK(bm_seekoff_cur<false>)->Name("std::ifstream::seekg(N, std::ios::cur) (unprimed buffer)")->Apply(run_sizes);

template <bool prime>
static void bm_seekoff_beg(benchmark::State& state) {
  std::vector<char> buffer;
  buffer.resize(16384);

  {
    std::ofstream gen_testfile("testfile");
    gen_testfile.write(buffer.data(), buffer.size());
  }

  std::ifstream stream("testfile");
  assert(stream);

  auto val = state.range();

  for (auto _ : state) {
    if constexpr (prime)
      benchmark::DoNotOptimize(stream.rdbuf()->sgetc());
    benchmark::DoNotOptimize(stream.seekg(val, std::ios::beg));
    benchmark::DoNotOptimize(stream.seekg(0, std::ios::beg));
  }
}
BENCHMARK(bm_seekoff_beg<true>)->Name("std::ifstream::seekg(N, std::ios::beg) (primed buffer)")->Apply(run_sizes);
BENCHMARK(bm_seekoff_beg<false>)->Name("std::ifstream::seekg(N, std::ios::beg) (unprimed buffer)")->Apply(run_sizes);

BENCHMARK_MAIN();
