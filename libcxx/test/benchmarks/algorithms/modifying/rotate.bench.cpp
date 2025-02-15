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
void bm(std::string operation_name, Operation rotate) {
  auto bench = [rotate](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    auto middle = std::next(c.begin(), size / 2);
    for ([[maybe_unused]] auto _ : st) {
      auto result = rotate(c.begin(), middle, c.end());
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_rotate    = [](auto first, auto middle, auto last) { return std::rotate(first, middle, last); };
  auto ranges_rotate = [](auto first, auto middle, auto last) { return std::ranges::rotate(first, middle, last); };

  // std::rotate
  bm<std::vector<int>>("std::rotate(vector<int>)", std_rotate);
  bm<std::deque<int>>("std::rotate(deque<int>)", std_rotate);
  bm<std::list<int>>("std::rotate(list<int>)", std_rotate);

  // ranges::rotate
  bm<std::vector<int>>("ranges::rotate(vector<int>)", ranges_rotate);
  bm<std::deque<int>>("ranges::rotate(deque<int>)", ranges_rotate);
  bm<std::list<int>>("ranges::rotate(list<int>)", ranges_rotate);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
