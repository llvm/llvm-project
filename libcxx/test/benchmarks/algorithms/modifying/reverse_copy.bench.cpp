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
  auto std_reverse_copy = [](auto first, auto last, auto out) { return std::reverse_copy(first, last, out); };

  // {std,ranges}::reverse_copy(normal container)
  {
    auto bm = []<class Container>(std::string name, auto reverse_copy) {
      benchmark::RegisterBenchmark(name, [reverse_copy](auto& st) {
        std::size_t const size = st.range(0);
        using ValueType        = typename Container::value_type;
        Container c;
        std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

        std::vector<ValueType> out(size);

        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(out);
          auto result = reverse_copy(c.begin(), c.end(), out.begin());
          benchmark::DoNotOptimize(result);
        }
      })->Range(8, 1 << 15);
    };
    bm.operator()<std::vector<int>>("std::reverse_copy(vector<int>)", std_reverse_copy);
    bm.operator()<std::deque<int>>("std::reverse_copy(deque<int>)", std_reverse_copy);
    bm.operator()<std::list<int>>("std::reverse_copy(list<int>)", std_reverse_copy);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
