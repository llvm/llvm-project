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
  auto std_adjacent_find      = [](auto first, auto last) { return std::adjacent_find(first, last); };
  auto std_adjacent_find_pred = [](auto first, auto last) {
    return std::adjacent_find(first, last, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };
  auto ranges_adjacent_find_pred = [](auto first, auto last) {
    return std::ranges::adjacent_find(first, last, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark {std,ranges}::adjacent_find on a sequence of the form xyxyxyxyxyxyxyxyxyxy,
  // which means we never find adjacent equal elements (the worst case of the algorithm).
  {
    auto bm = []<class Container>(std::string name, auto adjacent_find) {
      benchmark::RegisterBenchmark(
          name,
          [adjacent_find](auto& st) {
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
              auto result = adjacent_find(c.begin(), c.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };

    // {std,ranges}::adjacent_find
    bm.operator()<std::vector<int>>("std::adjacent_find(vector<int>)", std_adjacent_find);
    bm.operator()<std::deque<int>>("std::adjacent_find(deque<int>)", std_adjacent_find);
    bm.operator()<std::list<int>>("std::adjacent_find(list<int>)", std_adjacent_find);
    bm.operator()<std::vector<int>>("rng::adjacent_find(vector<int>)", std::ranges::adjacent_find);
    bm.operator()<std::deque<int>>("rng::adjacent_find(deque<int>)", std::ranges::adjacent_find);
    bm.operator()<std::list<int>>("rng::adjacent_find(list<int>)", std::ranges::adjacent_find);

    // {std,ranges}::adjacent_find(pred)
    bm.operator()<std::vector<int>>("std::adjacent_find(vector<int>, pred)", std_adjacent_find_pred);
    bm.operator()<std::deque<int>>("std::adjacent_find(deque<int>, pred)", std_adjacent_find_pred);
    bm.operator()<std::list<int>>("std::adjacent_find(list<int>, pred)", std_adjacent_find_pred);
    bm.operator()<std::vector<int>>("rng::adjacent_find(vector<int>, pred)", ranges_adjacent_find_pred);
    bm.operator()<std::deque<int>>("rng::adjacent_find(deque<int>, pred)", ranges_adjacent_find_pred);
    bm.operator()<std::list<int>>("rng::adjacent_find(list<int>, pred)", ranges_adjacent_find_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
