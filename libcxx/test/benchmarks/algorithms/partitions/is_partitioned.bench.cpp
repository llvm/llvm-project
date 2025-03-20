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

template <class Container, bool Partitioned, class Operation>
void bm(std::string operation_name, Operation is_partitioned) {
  auto bench = [is_partitioned](auto& st) {
    auto const size = st.range(0);
    using ValueType = typename Container::value_type;
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
      auto result = is_partitioned(c.begin(), c.end(), pred);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_is_partitioned    = [](auto first, auto last, auto pred) { return std::is_partitioned(first, last, pred); };
  auto ranges_is_partitioned = [](auto first, auto last, auto pred) {
    return std::ranges::is_partitioned(first, last, pred);
  };

  // std::is_partitioned
  bm<std::vector<int>, true>("std::is_partitioned(vector<int>) (partitioned)", std_is_partitioned);
  bm<std::vector<int>, false>("std::is_partitioned(vector<int>) (not partitioned)", std_is_partitioned);

  bm<std::deque<int>, true>("std::is_partitioned(deque<int>) (partitioned)", std_is_partitioned);
  bm<std::deque<int>, false>("std::is_partitioned(deque<int>) (not partitioned)", std_is_partitioned);

  bm<std::list<int>, true>("std::is_partitioned(list<int>) (partitioned)", std_is_partitioned);
  bm<std::list<int>, false>("std::is_partitioned(list<int>) (not partitioned)", std_is_partitioned);

  // ranges::is_partitioned
  bm<std::vector<int>, true>("ranges::is_partitioned(vector<int>) (partitioned)", ranges_is_partitioned);
  bm<std::vector<int>, false>("ranges::is_partitioned(vector<int>) (not partitioned)", ranges_is_partitioned);

  bm<std::deque<int>, true>("ranges::is_partitioned(deque<int>) (partitioned)", ranges_is_partitioned);
  bm<std::deque<int>, false>("ranges::is_partitioned(deque<int>) (not partitioned)", ranges_is_partitioned);

  bm<std::list<int>, true>("ranges::is_partitioned(list<int>) (partitioned)", ranges_is_partitioned);
  bm<std::list<int>, false>("ranges::is_partitioned(list<int>) (not partitioned)", ranges_is_partitioned);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
