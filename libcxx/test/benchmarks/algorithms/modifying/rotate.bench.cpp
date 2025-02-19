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
  auto std_rotate = [](auto first, auto middle, auto last) { return std::rotate(first, middle, last); };

  // {std,ranges}::rotate(normal container)
  {
    auto bm = []<class Container>(std::string name, auto rotate) {
      benchmark::RegisterBenchmark(
          name,
          [rotate](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

            auto middle = std::next(c.begin(), size / 2);
            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = rotate(c.begin(), middle, c.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::rotate(vector<int>)", std_rotate);
    bm.operator()<std::deque<int>>("std::rotate(deque<int>)", std_rotate);
    bm.operator()<std::list<int>>("std::rotate(list<int>)", std_rotate);
    bm.operator()<std::vector<int>>("rng::rotate(vector<int>)", std::ranges::rotate);
    bm.operator()<std::deque<int>>("rng::rotate(deque<int>)", std::ranges::rotate);
    bm.operator()<std::list<int>>("rng::rotate(list<int>)", std::ranges::rotate);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
