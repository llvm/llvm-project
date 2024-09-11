//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <random>
#include <vector>

struct remove_all {
  std::size_t operator()() { return 10; }
};

struct remove_first {
  std::size_t I = 10;
  std::size_t operator()() { return I++; }
};

struct remove_every_second {
  bool b = false;
  std::size_t operator()() {
    b = !b;
    return b ? 10 : 11;
  }
};

struct remove_random {
  std::shared_ptr<std::mt19937> rng = std::make_shared<std::mt19937>(std::random_device{}());
  std::size_t operator()() {
    return (*rng)();
  }
};

template <class T, class Generator>
static void bm_remove(benchmark::State& state) {
  std::vector<T> vec(state.range());
  Generator gen;
  std::generate(vec.begin(), vec.end(), gen);

  for (auto _ : state) {
    auto cpy = vec;
    benchmark::DoNotOptimize(cpy);
    benchmark::DoNotOptimize(std::remove(cpy.begin(), cpy.end(), char(10)));
  }
}
BENCHMARK(bm_remove<char, remove_all>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_remove<char, remove_first>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_remove<char, remove_every_second>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_remove<char, remove_random>)->DenseRange(1, 8)->Range(16, 1 << 20);

template <class T, class Generator>
static void bm_ranges_remove(benchmark::State& state) {
  std::vector<T> vec(state.range());
  Generator gen;
  std::generate(vec.begin(), vec.end(), gen);

  for (auto _ : state) {
    auto cpy = vec;
    benchmark::DoNotOptimize(cpy);
    benchmark::DoNotOptimize(std::ranges::remove(cpy, char(10)));
  }
}
BENCHMARK(bm_ranges_remove<char, remove_all>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_ranges_remove<char, remove_first>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_ranges_remove<char, remove_every_second>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_ranges_remove<char, remove_random>)->DenseRange(1, 8)->Range(16, 1 << 20);

BENCHMARK_MAIN();
