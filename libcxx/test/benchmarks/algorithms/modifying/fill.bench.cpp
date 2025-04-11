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
#include "test_macros.h"

int main(int argc, char** argv) {
  auto std_fill = [](auto first, auto last, auto const& value) { return std::fill(first, last, value); };

  // {std,ranges}::fill(normal container)
  {
    auto bm = []<class Container>(std::string name, auto fill) {
      benchmark::RegisterBenchmark(
          name,
          [fill](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            Container c(size, x);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(x);
              fill(c.begin(), c.end(), x);
              benchmark::DoNotOptimize(c);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::fill(vector<int>)", std_fill);
    bm.operator()<std::deque<int>>("std::fill(deque<int>)", std_fill);
    bm.operator()<std::list<int>>("std::fill(list<int>)", std_fill);
    bm.operator()<std::vector<int>>("rng::fill(vector<int>)", std::ranges::fill);
    bm.operator()<std::deque<int>>("rng::fill(deque<int>)", std::ranges::fill);
    bm.operator()<std::list<int>>("rng::fill(list<int>)", std::ranges::fill);
  }

  // {std,ranges}::fill(vector<bool>)
  {
    auto bm = [](std::string name, auto fill) {
      benchmark::RegisterBenchmark(name, [fill](auto& st) {
        std::size_t const size = st.range(0);
        bool x                 = true;
        std::vector<bool> c(size, x);

        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(x);
          fill(c.begin(), c.end(), x);
          benchmark::DoNotOptimize(c);
        }
      })->Range(64, 1 << 20);
    };
    bm("std::fill(vector<bool>)", std_fill);
#if TEST_STD_VER >= 23 // vector<bool>::iterator is not an output_iterator before C++23
    bm("rng::fill(vector<bool>)", std::ranges::fill);
#endif
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
