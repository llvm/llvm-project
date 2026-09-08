//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03

#include <cstdint>
#include <random>

#include <benchmark/benchmark.h>

template <typename Eng, std::uint64_t Max>
static void bm_uniform_int_distribution(benchmark::State& state) {
  Eng eng;
  std::uniform_int_distribution<std::uint64_t> dist(1ull, Max);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dist(eng));
  }
}

// n = 1
// Best Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, 1ull << 20>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, 1ull << 20>);
// Worst Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, (1ull << 19) + 1ull>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, (1ull << 19) + 1ull>);
// Median Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, (1ull << 19) + (1ull << 18)>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, (1ull << 19) + (1ull << 18)>);

// n = 2, n0 = 2
// Best Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, 1ull << 40>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, 1ull << 40>);
// Worst Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, (1ull << 39) + 1ull>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, (1ull << 39) + 1ull>);
// Median Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, (1ull << 39) + (1ull << 38)>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, (1ull << 39) + (1ull << 38)>);

// n = 2, n0 = 1
// Best Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, 1ull << 41>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, 1ull << 41>);
// Worst Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, (1ull << 40) + 1ull>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, (1ull << 40) + 1ull>);
// Median Case
BENCHMARK(bm_uniform_int_distribution<std::minstd_rand0, (1ull << 40) + (1ull << 39)>);
BENCHMARK(bm_uniform_int_distribution<std::ranlux24_base, (1ull << 40) + (1ull << 39)>);

BENCHMARK_MAIN();
