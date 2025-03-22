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
#define BENCH(generate_data, name)                                                                                     \
  do {                                                                                                                 \
    auto gen1 = [](auto size) { return generate_data<int>(size); };                                                    \
    auto gen2 = [](auto size) {                                                                                        \
      auto data = generate_data<int>(size);                                                                            \
      std::vector<support::NonIntegral> real_data(data.begin(), data.end());                                           \
      return real_data;                                                                                                \
    };                                                                                                                 \
    bm.operator()<std::vector<int>>("std::partial_sort_copy(vector<int>) (" #name ")", std_partial_sort_copy, gen1);   \
    bm.operator()<std::vector<support::NonIntegral>>(                                                                  \
        "std::partial_sort_copy(vector<NonIntegral>) (" #name ")", std_partial_sort_copy, gen2);                       \
    bm.operator()<std::deque<int>>("std::partial_sort_copy(deque<int>) (" #name ")", std_partial_sort_copy, gen1);     \
    bm.operator()<std::list<int>>("std::partial_sort_copy(list<int>) (" #name ")", std_partial_sort_copy, gen1);       \
                                                                                                                       \
    bm.operator()<std::vector<int>>(                                                                                   \
        "rng::partial_sort_copy(vector<int>) (" #name ")", std::ranges::partial_sort_copy, gen1);                      \
    bm.operator()<std::vector<support::NonIntegral>>(                                                                  \
        "rng::partial_sort_copy(vector<NonIntegral>) (" #name ")", std::ranges::partial_sort_copy, gen2);              \
    bm.operator()<std::deque<int>>(                                                                                    \
        "rng::partial_sort_copy(deque<int>) (" #name ")", std::ranges::partial_sort_copy, gen1);                       \
    bm.operator()<std::list<int>>(                                                                                     \
        "rng::partial_sort_copy(list<int>) (" #name ")", std::ranges::partial_sort_copy, gen1);                        \
  } while (false)

    BENCH(support::quicksort_adversarial_data, "qsort adversarial");
    BENCH(support::ascending_sorted_data, "ascending");
    BENCH(support::descending_sorted_data, "descending");
    BENCH(support::pipe_organ_data, "pipe-organ");
    BENCH(support::heap_data, "heap");
    BENCH(support::shuffled_data, "shuffled");
    BENCH(support::single_element_data, "repeated");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
