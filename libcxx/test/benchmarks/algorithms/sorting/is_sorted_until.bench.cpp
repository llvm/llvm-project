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
  auto std_is_sorted_until      = [](auto first, auto last) { return std::is_sorted_until(first, last); };
  auto std_is_sorted_until_pred = [](auto first, auto last) {
    return std::is_sorted_until(first, last, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x < y;
    });
  };
  auto ranges_is_sorted_until_pred = [](auto first, auto last) {
    return std::ranges::is_sorted_until(first, last, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x < y;
    });
  };

  // Benchmark {std,ranges}::is_sorted_until on a sorted sequence (the worst case).
  {
    auto bm = []<class Container>(std::string name, auto is_sorted_until) {
      benchmark::RegisterBenchmark(
          name,
          [is_sorted_until](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            std::vector<ValueType> data;
            std::generate_n(std::back_inserter(data), size, [] { return Generate<ValueType>::random(); });
            std::sort(data.begin(), data.end());

            Container c(data.begin(), data.end());

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = is_sorted_until(c.begin(), c.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::is_sorted_until(vector<int>)", std_is_sorted_until);
    bm.operator()<std::deque<int>>("std::is_sorted_until(deque<int>)", std_is_sorted_until);
    bm.operator()<std::list<int>>("std::is_sorted_until(list<int>)", std_is_sorted_until);
    bm.operator()<std::vector<int>>("rng::is_sorted_until(vector<int>)", std::ranges::is_sorted_until);
    bm.operator()<std::deque<int>>("rng::is_sorted_until(deque<int>)", std::ranges::is_sorted_until);
    bm.operator()<std::list<int>>("rng::is_sorted_until(list<int>)", std::ranges::is_sorted_until);

    bm.operator()<std::vector<int>>("std::is_sorted_until(vector<int>, pred)", std_is_sorted_until_pred);
    bm.operator()<std::deque<int>>("std::is_sorted_until(deque<int>, pred)", std_is_sorted_until_pred);
    bm.operator()<std::list<int>>("std::is_sorted_until(list<int>, pred)", std_is_sorted_until_pred);
    bm.operator()<std::vector<int>>("rng::is_sorted_until(vector<int>, pred)", ranges_is_sorted_until_pred);
    bm.operator()<std::deque<int>>("rng::is_sorted_until(deque<int>, pred)", ranges_is_sorted_until_pred);
    bm.operator()<std::list<int>>("rng::is_sorted_until(list<int>, pred)", ranges_is_sorted_until_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
