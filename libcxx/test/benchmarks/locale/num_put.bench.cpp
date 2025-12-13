//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <ios>
#include <locale>

#include <benchmark/benchmark.h>

struct num_put : std::num_put<char, std::string::iterator> {};

template <class T>
void BM_num_put(benchmark::State& state) {
  auto val = T(123);
  std::ios ios(nullptr);
  num_put np;

  for (auto _ : state) {
    benchmark::DoNotOptimize(val);
    std::string str;
    benchmark::DoNotOptimize(np.put(str.begin(), ios, ' ', val));
  }
}
BENCHMARK(BM_num_put<bool>);
BENCHMARK(BM_num_put<long>);
BENCHMARK(BM_num_put<long long>);
BENCHMARK(BM_num_put<unsigned long>);
BENCHMARK(BM_num_put<unsigned long long>);
BENCHMARK(BM_num_put<double>);
BENCHMARK(BM_num_put<long double>);
BENCHMARK(BM_num_put<const void*>);

BENCHMARK_MAIN();
