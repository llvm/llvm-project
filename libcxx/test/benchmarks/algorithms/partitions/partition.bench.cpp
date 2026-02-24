//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <iterator>
#include <list>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

auto compute_median(auto first, auto last) {
  std::vector v(first, last);
  auto middle = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), middle, v.end());
  return *middle;
}

int main(int argc, char** argv) {
  auto std_partition = [](auto first, auto last, auto pred) { return std::partition(first, last, pred); };

  // Benchmark {std,ranges}::partition on a fully unpartitionned sequence, i.e. a lot of elements
  // have to be moved around in order to partition the range.
  {
    auto bm = []<class Container>(std::string name, auto partition) {
      benchmark::RegisterBenchmark(
          name,
          [partition](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

            ValueType median = compute_median(c.begin(), c.end());
            auto pred1       = [median](auto const& element) { return element < median; };
            auto pred2       = [median](auto const& element) { return element > median; };
            bool toggle      = false;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              // By toggling the predicate, we have to move almost all elements in the sequence
              // to restore the partition.
              if (toggle) {
                auto result = partition(c.begin(), c.end(), pred1);
                benchmark::DoNotOptimize(result);
              } else {
                auto result = partition(c.begin(), c.end(), pred2);
                benchmark::DoNotOptimize(result);
              }
              toggle = !toggle;
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };

    // std::partition
    bm.operator()<std::vector<int>>("std::partition(vector<int>) (dense)", std_partition);
    bm.operator()<std::deque<int>>("std::partition(deque<int>) (dense)", std_partition);
    bm.operator()<std::list<int>>("std::partition(list<int>) (dense)", std_partition);
  }

  // Benchmark {std,ranges}::partition on a mostly partitioned sequence, i.e. only 10% of the elements
  // have to be moved around in order to partition the range.
  {
    auto bm = []<class Container>(std::string name, auto partition) {
      benchmark::RegisterBenchmark(
          name,
          [partition](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });
            ValueType median = compute_median(c.begin(), c.end());
            auto pred        = [median](auto const& element) { return element < median; };
            std::partition(c.begin(), c.end(), pred);

            // Between iterations, we swap 5% of the elements to the left of the median with 5% of the elements
            // to the right of the median. This ensures that the range is slightly unpartitioned.
            auto median_it = std::partition_point(c.begin(), c.end(), pred);
            auto low       = std::next(c.begin(), std::distance(c.begin(), median_it) - (size / 20));
            auto high      = std::next(median_it, size / 20);
            auto shuffle   = [&] { std::swap_ranges(low, median_it, high); };
            shuffle();
            assert(!std::is_partitioned(c.begin(), c.end(), pred));

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = partition(c.begin(), c.end(), pred);
              benchmark::DoNotOptimize(result);
              shuffle();
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };

    // std::partition
    bm.operator()<std::vector<int>>("std::partition(vector<int>) (sparse)", std_partition);
    bm.operator()<std::deque<int>>("std::partition(deque<int>) (sparse)", std_partition);
    bm.operator()<std::list<int>>("std::partition(list<int>) (sparse)", std_partition);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
