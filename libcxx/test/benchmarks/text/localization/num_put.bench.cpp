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
#include "test_macros.h"

struct num_put : std::num_put<char, std::string::iterator> {};

class grouping_numpunct : public std::numpunct<char> {
public:
  grouping_numpunct() : std::numpunct<char>() {}

protected:
  virtual char_type do_thousands_sep() const { return '_'; }
  virtual std::string do_grouping() const { return std::string("\1\2\3"); }
};

template <class NumPunct, class T>
TEST_ALIGN_BENCHMARK void BM_num_put(benchmark::State& state) {
  auto val = T(123);
  std::ios ios(nullptr);
  ios.imbue(std::locale(std::locale::classic(), new NumPunct));
  ios.setf(static_cast<std::ios::fmtflags>(state.range()));
  num_put np;

  std::string str('x', 10);
  for (auto _ : state) {
    benchmark::DoNotOptimize(val);
    benchmark::DoNotOptimize(str);
    if (state.range() != 0)
      ios.width(10);
    benchmark::DoNotOptimize(np.put(str.begin(), ios, ' ', val));
  }
}

void bench_padding(benchmark::Benchmark* bm) {
  bm->Arg(0)->Arg(std::ios::internal)->Arg(std::ios::left)->Arg(std::ios::right);
}

BENCHMARK(BM_num_put<std::numpunct<char>, bool>)->Apply(bench_padding);
BENCHMARK(BM_num_put<std::numpunct<char>, long>)->Apply(bench_padding);
BENCHMARK(BM_num_put<std::numpunct<char>, long long>)->Apply(bench_padding);
BENCHMARK(BM_num_put<std::numpunct<char>, unsigned long>)->Apply(bench_padding);
BENCHMARK(BM_num_put<std::numpunct<char>, unsigned long long>)->Apply(bench_padding);
BENCHMARK(BM_num_put<std::numpunct<char>, double>)->Apply(bench_padding);
BENCHMARK(BM_num_put<std::numpunct<char>, long double>)->Apply(bench_padding);
BENCHMARK(BM_num_put<std::numpunct<char>, const void*>)->Apply(bench_padding);

BENCHMARK(BM_num_put<grouping_numpunct, bool>)->Apply(bench_padding);
BENCHMARK(BM_num_put<grouping_numpunct, long>)->Apply(bench_padding);
BENCHMARK(BM_num_put<grouping_numpunct, long long>)->Apply(bench_padding);
BENCHMARK(BM_num_put<grouping_numpunct, unsigned long>)->Apply(bench_padding);
BENCHMARK(BM_num_put<grouping_numpunct, unsigned long long>)->Apply(bench_padding);
BENCHMARK(BM_num_put<grouping_numpunct, double>)->Apply(bench_padding);
BENCHMARK(BM_num_put<grouping_numpunct, long double>)->Apply(bench_padding);
BENCHMARK(BM_num_put<grouping_numpunct, const void*>)->Apply(bench_padding);

BENCHMARK_MAIN();
