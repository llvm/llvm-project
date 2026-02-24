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

#include "benchmark/benchmark.h"
#include "common.h"

int main(int argc, char** argv) {
  auto std_partial_sort_copy = [](auto first, auto last, auto dfirst, auto dlast) {
    return std::partial_sort_copy(first, last, dfirst, dlast);
  };

  // Benchmark {std,ranges}::partial_sort_copy on various types of data. We always partially
  // sort only half of the full range.
  //
  // Also note that we intentionally don't benchmark the predicated version of the algorithm
  // because that makes the benchmark run too slowly.
  {
    auto bm = []<class Container>(std::string name, auto partial_sort_copy, auto generate_data) {
      benchmark::RegisterBenchmark(
          name,
          [partial_sort_copy, generate_data](auto& st) {
            std::size_t const size      = st.range(0);
            using ValueType             = typename Container::value_type;
            std::vector<ValueType> data = generate_data(size);
            Container c(data.begin(), data.end());
            std::vector<ValueType> out(size / 2);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(out);
              auto result = partial_sort_copy(c.begin(), c.end(), out.begin(), out.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1024)
          ->Arg(8192);
    };

    auto register_bm = [&](auto generate, std::string variant) {
      auto gen2 = [generate](auto size) {
        std::vector<int> data = generate(size);
        std::vector<support::NonIntegral> real_data(data.begin(), data.end());
        return real_data;
      };
      auto name = [variant](std::string op) { return op + " (" + variant + ")"; };
      bm.operator()<std::vector<int>>(name("std::partial_sort_copy(vector<int>)"), std_partial_sort_copy, generate);
      bm.operator()<std::vector<support::NonIntegral>>(
          name("std::partial_sort_copy(vector<NonIntegral>)"), std_partial_sort_copy, gen2);
      bm.operator()<std::deque<int>>(name("std::partial_sort_copy(deque<int>)"), std_partial_sort_copy, generate);
      bm.operator()<std::list<int>>(name("std::partial_sort_copy(list<int>)"), std_partial_sort_copy, generate);
    };

    register_bm(support::quicksort_adversarial_data<int>, "qsort adversarial");
    register_bm(support::ascending_sorted_data<int>, "ascending");
    register_bm(support::descending_sorted_data<int>, "descending");
    register_bm(support::pipe_organ_data<int>, "pipe-organ");
    register_bm(support::heap_data<int>, "heap");
    register_bm(support::shuffled_data<int>, "shuffled");
    register_bm(support::single_element_data<int>, "repeated");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
