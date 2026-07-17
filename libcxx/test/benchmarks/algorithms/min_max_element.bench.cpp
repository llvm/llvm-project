//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <vector>

#include <benchmark/benchmark.h>

void run_sizes(benchmark::Benchmark* benchmark) { benchmark->Arg(2)->Arg(5500)->Arg(70000); }

template <class T>
void BM_std_minmax_element(benchmark::State& state) {
  std::vector<T> vec(state.range(), 3);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec);
    benchmark::DoNotOptimize(std::minmax_element(vec.begin(), vec.end()));
  }
}

BENCHMARK(BM_std_minmax_element<char>)->Apply(run_sizes);
BENCHMARK(BM_std_minmax_element<short>)->Apply(run_sizes);
BENCHMARK(BM_std_minmax_element<int>)->Apply(run_sizes);
BENCHMARK(BM_std_minmax_element<long long>)->Apply(run_sizes);

BENCHMARK_MAIN();
