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
  auto std_is_partitioned = [](auto first, auto last, auto pred) { return std::is_partitioned(first, last, pred); };

  auto bm = []<class Container, bool Partitioned>(std::string name, auto is_partitioned) {
    benchmark::RegisterBenchmark(
        name,
        [is_partitioned](auto& st) {
          std::size_t const size = st.range(0);
          using ValueType        = typename Container::value_type;
          Container c;
          std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

          // Partition the container in two equally-sized halves, ensuring the median
          // value appears in the left half. Note that the median value isn't located
          // in the middle -- this isn't std::nth_element.
          ValueType median = compute_median(c.begin(), c.end());
          auto pred        = [median](auto const& element) { return element <= median; };
          std::partition(c.begin(), c.end(), pred);
          assert(std::is_partitioned(c.begin(), c.end(), pred));

          if constexpr (!Partitioned) {
            // De-partition the container by swapping the element containing the median
            // value with the last one.
            auto median_it = std::find(c.begin(), c.end(), median);
            auto last_it   = std::next(c.begin(), c.size() - 1);
            std::iter_swap(median_it, last_it);
            assert(!std::is_partitioned(c.begin(), c.end(), pred));
          }

          for ([[maybe_unused]] auto _ : st) {
            benchmark::DoNotOptimize(c);
            auto result = is_partitioned(c.begin(), c.end(), pred);
            benchmark::DoNotOptimize(result);
          }
        })
        ->Arg(32)
        ->Arg(50) // non power-of-two
        ->Arg(1024)
        ->Arg(8192);
  };

  // std::is_partitioned
  bm.operator()<std::vector<int>, true>("std::is_partitioned(vector<int>) (partitioned)", std_is_partitioned);
  bm.operator()<std::vector<int>, false>("std::is_partitioned(vector<int>) (unpartitioned)", std_is_partitioned);

  bm.operator()<std::deque<int>, true>("std::is_partitioned(deque<int>) (partitioned)", std_is_partitioned);
  bm.operator()<std::deque<int>, false>("std::is_partitioned(deque<int>) (unpartitioned)", std_is_partitioned);

  bm.operator()<std::list<int>, true>("std::is_partitioned(list<int>) (partitioned)", std_is_partitioned);
  bm.operator()<std::list<int>, false>("std::is_partitioned(list<int>) (unpartitioned)", std_is_partitioned);

  // ranges::is_partitioned
  bm.operator()<std::vector<int>, true>("rng::is_partitioned(vector<int>) (partitioned)", std::ranges::is_partitioned);
  bm.operator()<std::vector<int>, false>(
      "rng::is_partitioned(vector<int>) (unpartitioned)", std::ranges::is_partitioned);

  bm.operator()<std::deque<int>, true>("rng::is_partitioned(deque<int>) (partitioned)", std::ranges::is_partitioned);
  bm.operator()<std::deque<int>, false>("rng::is_partitioned(deque<int>) (unpartitioned)", std::ranges::is_partitioned);

  bm.operator()<std::list<int>, true>("rng::is_partitioned(list<int>) (partitioned)", std::ranges::is_partitioned);
  bm.operator()<std::list<int>, false>("rng::is_partitioned(list<int>) (unpartitioned)", std::ranges::is_partitioned);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
