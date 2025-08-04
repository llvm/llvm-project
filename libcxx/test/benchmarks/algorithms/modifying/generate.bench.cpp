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
  auto std_generate = [](auto first, auto last, auto f) { return std::generate(first, last, f); };

  // {std,ranges}::generate
  {
    auto bm = []<class Container>(std::string name, auto generate) {
      benchmark::RegisterBenchmark(
          name,
          [generate](auto& st) {
            std::size_t const size = st.range(0);
            Container c(size);
            using ValueType = typename Container::value_type;
            ValueType x     = Generate<ValueType>::random();

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              generate(c.begin(), c.end(), [&x] {
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
    bm.operator()<std::vector<int>>("std::generate(vector<int>)", std_generate);
    bm.operator()<std::deque<int>>("std::generate(deque<int>)", std_generate);
    bm.operator()<std::list<int>>("std::generate(list<int>)", std_generate);
    bm.operator()<std::vector<int>>("rng::generate(vector<int>)", std::ranges::generate);
    bm.operator()<std::deque<int>>("rng::generate(deque<int>)", std::ranges::generate);
    bm.operator()<std::list<int>>("rng::generate(list<int>)", std::ranges::generate);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
