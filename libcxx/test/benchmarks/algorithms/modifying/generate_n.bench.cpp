//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <list>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_generate_n = [](auto out, auto n, auto f) { return std::generate_n(out, n, f); };

  // {std,ranges}::generate_n
  {
    auto bm = []<class Container>(std::string name, auto generate_n) {
      benchmark::RegisterBenchmark(
          name,
          [generate_n](auto& st) {
            std::size_t const size = st.range(0);
            Container c(size);
            using ValueType = typename Container::value_type;
            ValueType x     = Generate<ValueType>::random();

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              generate_n(c.begin(), size, [&x] {
                benchmark::DoNotOptimize(x);
                return x;
              });
              benchmark::DoNotOptimize(c);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::generate_n(vector<int>)", std_generate_n);
    bm.operator()<std::deque<int>>("std::generate_n(deque<int>)", std_generate_n);
    bm.operator()<std::list<int>>("std::generate_n(list<int>)", std_generate_n);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
