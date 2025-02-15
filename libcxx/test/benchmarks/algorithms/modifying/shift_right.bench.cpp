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
void bm(std::string operation_name, Operation shift_right) {
  auto bench = [shift_right](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    auto const n = 9 * (size / 10); // shift all but 10% of the range

    for ([[maybe_unused]] auto _ : st) {
      auto result = shift_right(c.begin(), c.end(), n);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_shift_right = [](auto first, auto last, auto n) { return std::shift_right(first, last, n); };

  // std::shift_right
  bm<std::vector<int>>("std::shift_right(vector<int>)", std_shift_right);
  bm<std::deque<int>>("std::shift_right(deque<int>)", std_shift_right);
  bm<std::list<int>>("std::shift_right(list<int>)", std_shift_right);

  // ranges::shift_right not implemented yet

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
