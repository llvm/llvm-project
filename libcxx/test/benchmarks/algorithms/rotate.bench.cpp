//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cassert>

#include <benchmark/benchmark.h>

void run_sizes(auto benchmark) {
  benchmark->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(32)->Arg(64)->Arg(512)->Arg(4096)->Arg(65536);
}

template <class T>
static void BM_std_rotate(benchmark::State& state) {
  std::vector<T> vec(state.range(), T());

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec);
    benchmark::DoNotOptimize(std::rotate(vec.begin(), vec.begin() + vec.size() / 3, vec.end()));
  }
}
BENCHMARK(BM_std_rotate<int>)->Apply(run_sizes);
#ifndef TEST_HAS_NO_INT128
BENCHMARK(BM_std_rotate<__int128>)->Apply(run_sizes);
#endif
BENCHMARK(BM_std_rotate<std::string>)->Apply(run_sizes);

BENCHMARK_MAIN();
