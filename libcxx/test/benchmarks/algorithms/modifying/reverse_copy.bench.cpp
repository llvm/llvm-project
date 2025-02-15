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
void bm(std::string operation_name, Operation reverse_copy) {
  auto bench = [reverse_copy](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> out(size);

    for ([[maybe_unused]] auto _ : st) {
      reverse_copy(c.begin(), c.end(), out.begin());
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Range(8, 1 << 15);
}

int main(int argc, char** argv) {
  auto std_reverse_copy    = [](auto first, auto last, auto out) { return std::reverse_copy(first, last, out); };
  auto ranges_reverse_copy = [](auto first, auto last, auto out) {
    return std::ranges::reverse_copy(first, last, out);
  };

  // std::reverse_copy
  bm<std::vector<int>>("std::reverse_copy(vector<int>)", std_reverse_copy);
  bm<std::deque<int>>("std::reverse_copy(deque<int>)", std_reverse_copy);
  bm<std::list<int>>("std::reverse_copy(list<int>)", std_reverse_copy);

  // ranges::reverse_copy
  bm<std::vector<int>>("ranges::reverse_copy(vector<int>)", ranges_reverse_copy);
  bm<std::deque<int>>("ranges::reverse_copy(deque<int>)", ranges_reverse_copy);
  bm<std::list<int>>("ranges::reverse_copy(list<int>)", ranges_reverse_copy);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
