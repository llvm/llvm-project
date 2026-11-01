//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <ios>
#include <locale>

#include <benchmark/benchmark.h>

#include "test_macros.h"

struct num_put : std::num_put<char, std::string::iterator> {};

class grouping_numpunct : public std::numpunct<char> {
public:
  grouping_numpunct() : std::numpunct<char>() {}

protected:
  char_type do_thousands_sep() const override { return '_'; }
  std::string do_grouping() const override { return std::string("\1\2\3"); }
};

template <class NumPunct, class IntT>
void bm(std::string name) {
  for (auto [flag, flag_name] :
       {std::pair{std::ios::fmtflags(), "none"},
        std::pair{std::ios::internal, "internal"},
        std::pair{std::ios::left, "left"},
        std::pair{std::ios::right, "right"}}) {
    benchmark::RegisterBenchmark(
        name + " (std::ios::" + flag_name + ')', [flag](benchmark::State& state) TEST_ALIGN_BENCHMARK {
          auto val = IntT(123);
          std::ios ios(nullptr);
          ios.imbue(std::locale(std::locale::classic(), new NumPunct));
          ios.setf(flag);
          num_put np;

          std::string str('x', 10);
          for (auto _ : state) {
            benchmark::DoNotOptimize(val);
            benchmark::DoNotOptimize(str);
            if (flag != 0)
              ios.width(10);
            benchmark::DoNotOptimize(np.put(str.begin(), ios, ' ', val));
          }
        });
  }
};

int main(int argc, char** argv) {
  bm<std::numpunct<char>, bool>("std::num_put::put(bool)");
  bm<std::numpunct<char>, long>("std::num_put::put(long)");
  bm<std::numpunct<char>, long long>("std::num_put::put(long long)");
  bm<std::numpunct<char>, unsigned long>("std::num_put::put(unsigned long)");
  bm<std::numpunct<char>, unsigned long long>("std::num_put::put(unsigned long long)");
  bm<std::numpunct<char>, double>("std::num_put::put(double)");
  bm<std::numpunct<char>, long double>("std::num_put::put(long double)");
  bm<std::numpunct<char>, const void*>("std::num_put::put(const void*)");

  bm<grouping_numpunct, bool>("std::num_put::put(bool) (grouping numpunct)");
  bm<grouping_numpunct, long>("std::num_put::put(long) (grouping numpunct)");
  bm<grouping_numpunct, long long>("std::num_put::put(long long) (grouping numpunct)");
  bm<grouping_numpunct, unsigned long>("std::num_put::put(unsigned long) (grouping numpunct)");
  bm<grouping_numpunct, unsigned long long>("std::num_put::put(unsigned long long) (grouping numpunct)");
  bm<grouping_numpunct, double>("std::num_put::put(double) (grouping numpunct)");
  bm<grouping_numpunct, long double>("std::num_put::put(long double) (grouping numpunct)");
  bm<grouping_numpunct, const void*>("std::num_put::put(const void*) (grouping numpunct)");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
