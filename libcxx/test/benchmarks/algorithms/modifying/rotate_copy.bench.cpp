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
void bm(std::string operation_name, Operation rotate_copy) {
  auto bench = [rotate_copy](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> out(size);

    auto middle = std::next(c.begin(), size / 2);
    for ([[maybe_unused]] auto _ : st) {
      auto result = rotate_copy(c.begin(), middle, c.end(), out.begin());
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_rotate_copy = [](auto first, auto middle, auto last, auto out) {
    return std::rotate_copy(first, middle, last, out);
  };
  auto ranges_rotate_copy = [](auto first, auto middle, auto last, auto out) {
    return std::ranges::rotate_copy(first, middle, last, out);
  };

  // std::rotate_copy
  bm<std::vector<int>>("std::rotate_copy(vector<int>)", std_rotate_copy);
  bm<std::deque<int>>("std::rotate_copy(deque<int>)", std_rotate_copy);
  bm<std::list<int>>("std::rotate_copy(list<int>)", std_rotate_copy);

  // ranges::rotate_copy
  bm<std::vector<int>>("ranges::rotate_copy(vector<int>)", ranges_rotate_copy);
  bm<std::deque<int>>("ranges::rotate_copy(deque<int>)", ranges_rotate_copy);
  bm<std::list<int>>("ranges::rotate_copy(list<int>)", ranges_rotate_copy);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
