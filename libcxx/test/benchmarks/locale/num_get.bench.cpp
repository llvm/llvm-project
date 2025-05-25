
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

struct num_get : std::num_get<char, std::string::iterator> {};

template <class T>
void BM_num_get(benchmark::State& state) {
  auto val = std::string("123");
  std::ios ios(nullptr);
  num_get np;

  for (auto _ : state) {
    benchmark::DoNotOptimize(val);
    T out;
    std::ios_base::iostate err = ios.goodbit;
    benchmark::DoNotOptimize(np.get(val.begin(), val.end(), ios, err, out));
    benchmark::DoNotOptimize(out);
  }
}

BENCHMARK(BM_num_get<bool>);
BENCHMARK(BM_num_get<long>);
BENCHMARK(BM_num_get<long long>);
BENCHMARK(BM_num_get<unsigned short>);
BENCHMARK(BM_num_get<unsigned int>);
BENCHMARK(BM_num_get<unsigned long>);
BENCHMARK(BM_num_get<unsigned long long>);
BENCHMARK(BM_num_get<float>);
BENCHMARK(BM_num_get<double>);
BENCHMARK(BM_num_get<long double>);
BENCHMARK(BM_num_get<void*>);

BENCHMARK_MAIN();
