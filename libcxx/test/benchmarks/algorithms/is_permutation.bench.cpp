//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <benchmark/benchmark.h>

void BenchmarkSizes(benchmark::internal::Benchmark* Benchmark) {
  Benchmark->DenseRange(1, 8);
  for (size_t i = 16; i != 1 << 20; i *= 2) {
    Benchmark->Arg(i - 1);
    Benchmark->Arg(i);
    Benchmark->Arg(i + 1);
  }
}

// Test std::is_permutation when sequences are identical
static void bm_std_is_permutation_same(benchmark::State& state) {
  std::vector<int> vec1(state.range(), 1);
  std::vector<int> vec2(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::is_permutation(vec1.begin(), vec1.end(), vec2.begin()));
  }
}
BENCHMARK(bm_std_is_permutation_same)->Apply(BenchmarkSizes);

// Test std::ranges::is_permutation when sequences are identical
static void bm_ranges_is_permutation_same(benchmark::State& state) {
  std::vector<int> vec1(state.range(), 1);
  std::vector<int> vec2(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::ranges::is_permutation(vec1, vec2));
  }
}
BENCHMARK(bm_ranges_is_permutation_same)->Apply(BenchmarkSizes);

// Test std::is_permutation when sequences are permutations
static void bm_std_is_permutation_shuffled(benchmark::State& state) {
  std::vector<int> vec1(state.range());
  std::iota(vec1.begin(), vec1.end(), 0);
  auto vec2 = vec1;
  std::mt19937 gen(42);
  std::shuffle(vec2.begin(), vec2.end(), gen);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::is_permutation(vec1.begin(), vec1.end(), vec2.begin()));
  }
}
BENCHMARK(bm_std_is_permutation_shuffled)->Apply(BenchmarkSizes);

// Test std::ranges::is_permutation when sequences are permutations
static void bm_ranges_is_permutation_shuffled(benchmark::State& state) {
  std::vector<int> vec1(state.range());
  std::iota(vec1.begin(), vec1.end(), 0);
  auto vec2 = vec1;
  std::mt19937 gen(42);
  std::shuffle(vec2.begin(), vec2.end(), gen);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::ranges::is_permutation(vec1, vec2));
  }
}
BENCHMARK(bm_ranges_is_permutation_shuffled)->Apply(BenchmarkSizes);

// Test std::is_permutation when sequences differ in last element
static void bm_std_is_permutation_diff_last(benchmark::State& state) {
  std::vector<int> vec1(state.range(), 1);
  std::vector<int> vec2(state.range(), 1);
  vec2.back() = 2;

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::is_permutation(vec1.begin(), vec1.end(), vec2.begin()));
  }
}
BENCHMARK(bm_std_is_permutation_diff_last)->Apply(BenchmarkSizes);

// Test std::ranges::is_permutation when sequences differ in last element
static void bm_ranges_is_permutation_diff_last(benchmark::State& state) {
  std::vector<int> vec1(state.range(), 1);
  std::vector<int> vec2(state.range(), 1);
  vec2.back() = 2;

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(vec2);
    benchmark::DoNotOptimize(std::ranges::is_permutation(vec1, vec2));
  }
}
BENCHMARK(bm_ranges_is_permutation_diff_last)->Apply(BenchmarkSizes);

BENCHMARK_MAIN();