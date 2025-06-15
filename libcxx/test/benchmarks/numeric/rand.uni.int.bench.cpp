//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03

#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>

template <typename Eng, std::uint64_t Max>
static void bm_rand_uni_int(benchmark::State& state) {
  Eng eng;
  std::uniform_int_distribution<std::uint64_t> dist(1ull, Max);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dist(eng));
  }
}

// n = 1
BENCHMARK(bm_rand_uni_int<std::minstd_rand0, 1ull << 20>);
BENCHMARK(bm_rand_uni_int<std::ranlux24_base, 1ull << 20>);

// n = 2, n0 = 2
BENCHMARK(bm_rand_uni_int<std::minstd_rand0, 1ull << 40>);
BENCHMARK(bm_rand_uni_int<std::ranlux24_base, 1ull << 40>);

// n = 2, n0 = 1
BENCHMARK(bm_rand_uni_int<std::minstd_rand0, 1ull << 41>);
BENCHMARK(bm_rand_uni_int<std::ranlux24_base, 1ull << 41>);

BENCHMARK_MAIN();
