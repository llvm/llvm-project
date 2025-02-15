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
void bm(std::string operation_name, Operation reverse) {
  auto bench = [reverse](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    for ([[maybe_unused]] auto _ : st) {
      reverse(c.begin(), c.end());
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Range(8, 1 << 15);
}

int main(int argc, char** argv) {
  auto std_reverse    = [](auto first, auto last) { return std::reverse(first, last); };
  auto ranges_reverse = [](auto first, auto last) { return std::ranges::reverse(first, last); };

  // std::reverse
  bm<std::vector<int>>("std::reverse(vector<int>)", std_reverse);
  bm<std::deque<int>>("std::reverse(deque<int>)", std_reverse);
  bm<std::list<int>>("std::reverse(list<int>)", std_reverse);

  // ranges::reverse
  bm<std::vector<int>>("ranges::reverse(vector<int>)", ranges_reverse);
  bm<std::deque<int>>("ranges::reverse(deque<int>)", ranges_reverse);
  bm<std::list<int>>("ranges::reverse(list<int>)", ranges_reverse);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
