//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include "benchmark/benchmark.h"
#include <bitset>
#include <cmath>
#include <cstddef>
#include <random>

template <std::size_t N>
struct GenerateBitset {
  // Construct a bitset with N bits, where each bit is set with probability p.
  static std::bitset<N> generate(double p) {
    std::bitset<N> b;
    if (p <= 0.0)
      return b;
    if (p >= 1.0)
      return ~b;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(p);
    for (std::size_t i = 0; i < N; ++i)
      b[i] = d(gen);

    return b;
  }

  static std::bitset<N> sparse() { return generate(0.1); }
  static std::bitset<N> dense() { return generate(0.9); }
  static std::bitset<N> uniform() { return generate(0.5); }
};

template <std::size_t N>
static void BM_BitsetToString(benchmark::State& state) {
  double p         = state.range(0) / 100.0;
  std::bitset<N> b = GenerateBitset<N>::generate(p);
  benchmark::DoNotOptimize(b);

  for (auto _ : state) {
    benchmark::DoNotOptimize(b.to_string());
  }
}

// Sparse bitset
BENCHMARK(BM_BitsetToString<32>)->Arg(10)->Name("BM_BitsetToString<32>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<64>)->Arg(10)->Name("BM_BitsetToString<64>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<8192>)->Arg(10)->Name("BM_BitsetToString<8192>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<1048576>)->Arg(10)->Name("BM_BitsetToString<1048576>/Sparse (10%)"); // 1 << 20

// Dense bitset
BENCHMARK(BM_BitsetToString<32>)->Arg(90)->Name("BM_BitsetToString<32>/Dense (90%)");
BENCHMARK(BM_BitsetToString<64>)->Arg(90)->Name("BM_BitsetToString<64>/Dense (90%)");
BENCHMARK(BM_BitsetToString<8192>)->Arg(90)->Name("BM_BitsetToString<8192>/Dense (90%)");
BENCHMARK(BM_BitsetToString<1048576>)->Arg(90)->Name("BM_BitsetToString<1048576>/Dense (90%)"); // 1 << 20

// Uniform bitset
BENCHMARK(BM_BitsetToString<32>)->Arg(50)->Name("BM_BitsetToString<32>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<64>)->Arg(50)->Name("BM_BitsetToString<64>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<8192>)->Arg(50)->Name("BM_BitsetToString<8192>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<1048576>)->Arg(50)->Name("BM_BitsetToString<1048576>/Uniform (50%)"); // 1 << 20

static void BM_Bitset_ctor_ull(benchmark::State& state) {
  unsigned long long val = 1;
  for (auto _ : state) {
    benchmark::DoNotOptimize(val);
    std::bitset<128> b(val);
    benchmark::DoNotOptimize(b);
  }
}

BENCHMARK(BM_Bitset_ctor_ull);

BENCHMARK_MAIN();
