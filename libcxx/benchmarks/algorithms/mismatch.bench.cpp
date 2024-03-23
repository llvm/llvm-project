//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>

// TODO: Look into benchmarking aligned and unaligned memory explicitly
// (currently things happen to be aligned because they are malloced that way)
template <class T>
static void bm_mismatch(benchmark::State& state) {
  std::vector<T> vec1(state.range(), '1');
  std::vector<T> vec2(state.range(), '1');
  std::mt19937_64 rng(std::random_device{}());

  vec1.back() = '2';
  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(std::mismatch(vec1.begin(), vec1.end(), vec2.begin()));
  }
}
BENCHMARK(bm_mismatch<char>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_mismatch<short>)->DenseRange(1, 8)->Range(16, 1 << 20);
BENCHMARK(bm_mismatch<int>)->DenseRange(1, 8)->Range(16, 1 << 20);

BENCHMARK_MAIN();
