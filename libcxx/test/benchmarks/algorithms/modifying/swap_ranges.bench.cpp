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

template <class Container, class Operation>
void bm(std::string operation_name, Operation swap_ranges) {
  auto bench = [swap_ranges](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c1, c2;
    std::generate_n(std::back_inserter(c1), size, [] { return Generate<ValueType>::random(); });
    std::generate_n(std::back_inserter(c2), size, [] { return Generate<ValueType>::random(); });

    for ([[maybe_unused]] auto _ : st) {
      auto result = swap_ranges(c1.begin(), c1.end(), c2.begin(), c2.end());
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c1);
      benchmark::DoNotOptimize(c2);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_swap_ranges = [](auto first1, auto last1, auto first2, auto) {
    return std::swap_ranges(first1, last1, first2);
  };
  auto ranges_swap_ranges = [](auto first1, auto last1, auto first2, auto last2) {
    return std::ranges::swap_ranges(first1, last1, first2, last2);
  };

  // std::swap_ranges
  bm<std::vector<int>>("std::swap_ranges(vector<int>)", std_swap_ranges);
  bm<std::deque<int>>("std::swap_ranges(deque<int>)", std_swap_ranges);
  bm<std::list<int>>("std::swap_ranges(list<int>)", std_swap_ranges);

  // ranges::swap_ranges
  bm<std::vector<int>>("ranges::swap_ranges(vector<int>)", ranges_swap_ranges);
  bm<std::deque<int>>("ranges::swap_ranges(deque<int>)", ranges_swap_ranges);
  bm<std::list<int>>("ranges::swap_ranges(list<int>)", ranges_swap_ranges);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
