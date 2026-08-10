//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

#include "benchmark/benchmark.h"

constexpr std::size_t MAX_BUFFER_LEN = 256;
constexpr std::size_t MAX_SEED_LEN   = 16;

static void BM_SeedSeq_Generate(benchmark::State& state) {
  std::array<std::uint32_t, MAX_BUFFER_LEN> buffer;
  std::array<std::uint32_t, MAX_SEED_LEN> seeds;
  {
    std::random_device rd;
    std::generate(std::begin(seeds), std::begin(seeds) + state.range(0), [&]() { return rd(); });
  }
  std::seed_seq seed(std::begin(seeds), std::begin(seeds) + state.range(0));
  for (auto _ : state) {
    seed.generate(std::begin(buffer), std::begin(buffer) + state.range(1));
  }
}
BENCHMARK(BM_SeedSeq_Generate)->Ranges({{1, MAX_SEED_LEN}, {1, MAX_BUFFER_LEN}});

template <class Engine>
static void BM_engine(benchmark::State& state) {
  Engine engine;

  for (auto _ : state) {
    benchmark::DoNotOptimize(engine());
  }
}
BENCHMARK(BM_engine<std::mt19937_64>)->Name("std::mt19937_64::operator()");
BENCHMARK(BM_engine<std::mt19937>)->Name("std::mt19937::operator()");

template <class Engine>
static void BM_engine_array(benchmark::State& state) {
  Engine engine;

  typename Engine::result_type buffer[128];

  for (auto _ : state) {
    std::generate(std::begin(buffer), std::end(buffer), std::ref(engine));
    benchmark::DoNotOptimize(buffer);
  }
}
BENCHMARK(BM_engine_array<std::mt19937_64>)->Name("std::mt19937_64::operator() (into array)");
BENCHMARK(BM_engine_array<std::mt19937>)->Name("std::mt19937::operator() (into array)");

BENCHMARK_MAIN();
