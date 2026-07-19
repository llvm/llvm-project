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
  auto std_fill_n = [](auto out, auto n, auto const& value) { return std::fill_n(out, n, value); };

  // {std,ranges}::fill_n(normal container)
  {
    auto bm = []<class Container>(std::string name, auto fill_n) {
      benchmark::RegisterBenchmark(
          name,
          [fill_n](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            Container c(size, x);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(x);
              fill_n(c.begin(), size, x);
              benchmark::DoNotOptimize(c);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::fill_n(vector<int>)", std_fill_n);
    bm.operator()<std::deque<int>>("std::fill_n(deque<int>)", std_fill_n);
    bm.operator()<std::list<int>>("std::fill_n(list<int>)", std_fill_n);
  }

  // {std,ranges}::fill_n(vector<bool>)
  {
    auto bm = [](std::string name, auto fill_n) {
      benchmark::RegisterBenchmark(name, [fill_n](auto& st) {
        std::size_t const size = st.range(0);
        bool x                 = true;
        std::vector<bool> c(size, x);

        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(x);
          fill_n(c.begin(), size, x);
          benchmark::DoNotOptimize(c);
        }
      })->Range(64, 1 << 20);
    };
    bm("std::fill_n(vector<bool>)", std_fill_n);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
