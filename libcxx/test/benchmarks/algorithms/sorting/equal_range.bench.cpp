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
#include <forward_list>
#include <list>
#include <numeric>
#include <vector>

#include "benchmark/benchmark.h"

int main(int argc, char** argv) {
  auto std_equal_range = [](auto first, auto last, auto const& value) { return std::equal_range(first, last, value); };
  auto std_equal_range_pred = [](auto first, auto last, auto const& value) {
    return std::equal_range(first, last, value, [](auto x, auto y) { return x < y; });
  };

  // The range we find is a single element.
  {
    auto bm = []<class Container>(std::string name, auto equal_range) {
      benchmark::RegisterBenchmark(
          name,
          [equal_range](auto& st) {
            std::size_t const size = st.range(0);
            std::vector<int> data(size);
            std::iota(data.begin(), data.end(), 0);
            int const key = static_cast<int>(size / 2);

            Container c(data.begin(), data.end());
            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = equal_range(c.begin(), c.end(), key);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(100)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::equal_range(vector<int>) (unique)", std_equal_range);
    bm.operator()<std::deque<int>>("std::equal_range(deque<int>) (unique)", std_equal_range);
    bm.operator()<std::list<int>>("std::equal_range(list<int>) (unique)", std_equal_range);
    bm.operator()<std::forward_list<int>>("std::equal_range(forward_list<int>) (unique)", std_equal_range);
    bm.operator()<std::vector<int>>("std::equal_range(vector<int>, pred) (unique)", std_equal_range_pred);
    bm.operator()<std::deque<int>>("std::equal_range(deque<int>, pred) (unique)", std_equal_range_pred);
    bm.operator()<std::list<int>>("std::equal_range(list<int>, pred) (unique)", std_equal_range_pred);
    bm.operator()<std::forward_list<int>>("std::equal_range(forward_list<int>, pred) (unique)", std_equal_range_pred);
  }

  // The range we find is a large part of the entire range.
  // Data looks like [0, 1, 2, 2, 2, 2, 2, 7, 8, 9], we match the 2's.
  {
    auto bm = []<class Container>(std::string name, auto equal_range) {
      benchmark::RegisterBenchmark(
          name,
          [equal_range](auto& st) {
            std::size_t const size          = st.range(0);
            std::size_t const subrange_size = size / 2;
            std::size_t const left_flank    = (size - subrange_size) / 2;
            std::vector<int> data(size);
            std::iota(data.begin(), data.end(), 0); // [0, 1, 2, 3, ..., 8, 9]
            int const key = data[left_flank];
            std::fill_n(data.begin() + left_flank, subrange_size,
                        key); // [0, 1, 2, 3, 3, 3, 3, ..., 8, 9]

            Container c(data.begin(), data.end());
            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = equal_range(c.begin(), c.end(), key);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(100)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::equal_range(vector<int>) (large range)", std_equal_range);
    bm.operator()<std::deque<int>>("std::equal_range(deque<int>) (large range)", std_equal_range);
    bm.operator()<std::list<int>>("std::equal_range(list<int>) (large range)", std_equal_range);
    bm.operator()<std::forward_list<int>>("std::equal_range(forward_list<int>) (large range)", std_equal_range);
    bm.operator()<std::vector<int>>("std::equal_range(vector<int>, pred) (large range)", std_equal_range_pred);
    bm.operator()<std::deque<int>>("std::equal_range(deque<int>, pred) (large range)", std_equal_range_pred);
    bm.operator()<std::list<int>>("std::equal_range(list<int>, pred) (large range)", std_equal_range_pred);
    bm.operator()<std::forward_list<int>>(
        "std::equal_range(forward_list<int>, pred) (large range)", std_equal_range_pred);
  }

  // The searched-for value is not present, so the found range is empty.
  {
    auto bm = []<class Container>(std::string name, auto equal_range) {
      benchmark::RegisterBenchmark(
          name,
          [equal_range](auto& st) {
            std::size_t const size = st.range(0);
            std::vector<int> data(size);
            std::iota(data.begin(), data.end(), 0);
            int const key = static_cast<int>(size); // one past the last element

            Container c(data.begin(), data.end());
            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = equal_range(c.begin(), c.end(), key);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(100)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::equal_range(vector<int>) (absent)", std_equal_range);
    bm.operator()<std::deque<int>>("std::equal_range(deque<int>) (absent)", std_equal_range);
    bm.operator()<std::list<int>>("std::equal_range(list<int>) (absent)", std_equal_range);
    bm.operator()<std::forward_list<int>>("std::equal_range(forward_list<int>) (absent)", std_equal_range);
    bm.operator()<std::vector<int>>("std::equal_range(vector<int>, pred) (absent)", std_equal_range_pred);
    bm.operator()<std::deque<int>>("std::equal_range(deque<int>, pred) (absent)", std_equal_range_pred);
    bm.operator()<std::list<int>>("std::equal_range(list<int>, pred) (absent)", std_equal_range_pred);
    bm.operator()<std::forward_list<int>>("std::equal_range(forward_list<int>, pred) (absent)", std_equal_range_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
