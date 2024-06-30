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

void BenchmarkSizes(benchmark::internal::Benchmark* Benchmark) {
  Benchmark->DenseRange(1, 8);
  for (size_t i = 16; i != 1 << 20; i *= 2) {
    Benchmark->Arg(i - 1);
    Benchmark->Arg(i);
    Benchmark->Arg(i + 1);
  }
}

// TODO: Look into benchmarking aligned and unaligned memory explicitly
// (currently things happen to be aligned because they are malloced that way)
template <class T>
static void bm_adjacent_find(benchmark::State& state) {
  std::vector<T> vec1(state.range());

  size_t val = 1;
  for (auto& e : vec1) {
    e = val++;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec1);
    benchmark::DoNotOptimize(std::adjacent_find(vec1.begin(), vec1.end()));
  }
}
BENCHMARK(bm_adjacent_find<char>)->Apply(BenchmarkSizes);
BENCHMARK(bm_adjacent_find<short>)->Apply(BenchmarkSizes);
BENCHMARK(bm_adjacent_find<int>)->Apply(BenchmarkSizes);

BENCHMARK_MAIN();
