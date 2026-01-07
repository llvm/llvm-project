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
BENCHMARK(BM_BitsetToString<128>)->Arg(10)->Name("BM_BitsetToString<128>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<256>)->Arg(10)->Name("BM_BitsetToString<256>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<512>)->Arg(10)->Name("BM_BitsetToString<512>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<1024>)->Arg(10)->Name("BM_BitsetToString<1024>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<2048>)->Arg(10)->Name("BM_BitsetToString<2048>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<4096>)->Arg(10)->Name("BM_BitsetToString<4096>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<8192>)->Arg(10)->Name("BM_BitsetToString<8192>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<16384>)->Arg(10)->Name("BM_BitsetToString<16384>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<32768>)->Arg(10)->Name("BM_BitsetToString<32768>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<65536>)->Arg(10)->Name("BM_BitsetToString<65536>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<131072>)->Arg(10)->Name("BM_BitsetToString<131072>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<262144>)->Arg(10)->Name("BM_BitsetToString<262144>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<524288>)->Arg(10)->Name("BM_BitsetToString<524288>/Sparse (10%)");
BENCHMARK(BM_BitsetToString<1048576>)->Arg(10)->Name("BM_BitsetToString<1048576>/Sparse (10%)"); // 1 << 20

// Dense bitset
BENCHMARK(BM_BitsetToString<32>)->Arg(90)->Name("BM_BitsetToString<32>/Dense (90%)");
BENCHMARK(BM_BitsetToString<64>)->Arg(90)->Name("BM_BitsetToString<64>/Dense (90%)");
BENCHMARK(BM_BitsetToString<128>)->Arg(90)->Name("BM_BitsetToString<128>/Dense (90%)");
BENCHMARK(BM_BitsetToString<256>)->Arg(90)->Name("BM_BitsetToString<256>/Dense (90%)");
BENCHMARK(BM_BitsetToString<512>)->Arg(90)->Name("BM_BitsetToString<512>/Dense (90%)");
BENCHMARK(BM_BitsetToString<1024>)->Arg(90)->Name("BM_BitsetToString<1024>/Dense (90%)");
BENCHMARK(BM_BitsetToString<2048>)->Arg(90)->Name("BM_BitsetToString<2048>/Dense (90%)");
BENCHMARK(BM_BitsetToString<4096>)->Arg(90)->Name("BM_BitsetToString<4096>/Dense (90%)");
BENCHMARK(BM_BitsetToString<8192>)->Arg(90)->Name("BM_BitsetToString<8192>/Dense (90%)");
BENCHMARK(BM_BitsetToString<16384>)->Arg(90)->Name("BM_BitsetToString<16384>/Dense (90%)");
BENCHMARK(BM_BitsetToString<32768>)->Arg(90)->Name("BM_BitsetToString<32768>/Dense (90%)");
BENCHMARK(BM_BitsetToString<65536>)->Arg(90)->Name("BM_BitsetToString<65536>/Dense (90%)");
BENCHMARK(BM_BitsetToString<131072>)->Arg(90)->Name("BM_BitsetToString<131072>/Dense (90%)");
BENCHMARK(BM_BitsetToString<262144>)->Arg(90)->Name("BM_BitsetToString<262144>/Dense (90%)");
BENCHMARK(BM_BitsetToString<524288>)->Arg(90)->Name("BM_BitsetToString<524288>/Dense (90%)");
BENCHMARK(BM_BitsetToString<1048576>)->Arg(90)->Name("BM_BitsetToString<1048576>/Dense (90%)"); // 1 << 20

// Uniform bitset
BENCHMARK(BM_BitsetToString<32>)->Arg(50)->Name("BM_BitsetToString<32>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<64>)->Arg(50)->Name("BM_BitsetToString<64>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<128>)->Arg(50)->Name("BM_BitsetToString<128>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<256>)->Arg(50)->Name("BM_BitsetToString<256>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<512>)->Arg(50)->Name("BM_BitsetToString<512>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<1024>)->Arg(50)->Name("BM_BitsetToString<1024>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<2048>)->Arg(50)->Name("BM_BitsetToString<2048>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<4096>)->Arg(50)->Name("BM_BitsetToString<4096>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<8192>)->Arg(50)->Name("BM_BitsetToString<8192>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<16384>)->Arg(50)->Name("BM_BitsetToString<16384>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<32768>)->Arg(50)->Name("BM_BitsetToString<32768>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<65536>)->Arg(50)->Name("BM_BitsetToString<65536>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<131072>)->Arg(50)->Name("BM_BitsetToString<131072>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<262144>)->Arg(50)->Name("BM_BitsetToString<262144>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<524288>)->Arg(50)->Name("BM_BitsetToString<524288>/Uniform (50%)");
BENCHMARK(BM_BitsetToString<1048576>)->Arg(50)->Name("BM_BitsetToString<1048576>/Uniform (50%)"); // 1 << 20

static void BM_Bitset_ctor_ull(benchmark::State& state) {
  unsigned long long val = (1ULL << state.range(0)) - 1;
  for (auto _ : state) {
    std::bitset<128> b(val);
    benchmark::DoNotOptimize(b);
  }
}

BENCHMARK(BM_Bitset_ctor_ull)->DenseRange(1, 63);

BENCHMARK_MAIN();
