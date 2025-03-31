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

int main(int argc, char** argv) {
  auto std_reverse = [](auto first, auto last) { return std::reverse(first, last); };

  // {std,ranges}::reverse(normal container)
  {
    auto bm = []<class Container>(std::string name, auto reverse) {
      benchmark::RegisterBenchmark(name, [reverse](auto& st) {
        std::size_t const size = st.range(0);
        using ValueType        = typename Container::value_type;
        Container c;
        std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c);
          reverse(c.begin(), c.end());
          benchmark::DoNotOptimize(c);
        }
      })->Range(8, 1 << 15);
    };
    bm.operator()<std::vector<int>>("std::reverse(vector<int>)", std_reverse);
    bm.operator()<std::deque<int>>("std::reverse(deque<int>)", std_reverse);
    bm.operator()<std::list<int>>("std::reverse(list<int>)", std_reverse);
    bm.operator()<std::vector<int>>("rng::reverse(vector<int>)", std::ranges::reverse);
    bm.operator()<std::deque<int>>("rng::reverse(deque<int>)", std::ranges::reverse);
    bm.operator()<std::list<int>>("rng::reverse(list<int>)", std::ranges::reverse);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
