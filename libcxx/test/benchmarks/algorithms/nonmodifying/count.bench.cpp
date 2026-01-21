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
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_count    = [](auto first, auto last, auto const& value) { return std::count(first, last, value); };
  auto std_count_if = [](auto first, auto last, auto const& value) {
    return std::count_if(first, last, [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element == value;
    });
  };

  // Benchmark {std,ranges}::{count,count_if} on a sequence where every other element is counted.
  {
    auto bm = []<class Container>(std::string name, auto count) {
      benchmark::RegisterBenchmark(
          name,
          [count](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c;
            for (std::size_t i = 0; i != size; ++i) {
              c.push_back(i % 2 == 0 ? x : y);
            }

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(x);
              auto result = count(c.begin(), c.end(), x);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };

    // count
    bm.operator()<std::vector<int>>("std::count(vector<int>) (every other)", std_count);
    bm.operator()<std::deque<int>>("std::count(deque<int>) (every other)", std_count);
    bm.operator()<std::list<int>>("std::count(list<int>) (every other)", std_count);

    // count_if
    bm.operator()<std::vector<int>>("std::count_if(vector<int>) (every other)", std_count_if);
    bm.operator()<std::deque<int>>("std::count_if(deque<int>) (every other)", std_count_if);
    bm.operator()<std::list<int>>("std::count_if(list<int>) (every other)", std_count_if);
  }

  // Benchmark {std,ranges}::count(vector<bool>)
  {
    auto bm = [](std::string name, auto count) {
      benchmark::RegisterBenchmark(
          name,
          [count](auto& st) {
            std::size_t const size = st.range(0);
            std::vector<bool> c(size, false);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = count(c.begin(), c.end(), true);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()("std::count(vector<bool>)", std_count);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
