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
  auto std_rotate_copy = [](auto first, auto middle, auto last, auto out) {
    return std::rotate_copy(first, middle, last, out);
  };

  // {std,ranges}::rotate_copy(normal container)
  {
    auto bm = []<class Container>(std::string name, auto rotate_copy) {
      benchmark::RegisterBenchmark(
          name,
          [rotate_copy](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

            std::vector<ValueType> out(size);

            auto middle = std::next(c.begin(), size / 2);
            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(out);
              auto result = rotate_copy(c.begin(), middle, c.end(), out.begin());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::rotate_copy(vector<int>)", std_rotate_copy);
    bm.operator()<std::deque<int>>("std::rotate_copy(deque<int>)", std_rotate_copy);
    bm.operator()<std::list<int>>("std::rotate_copy(list<int>)", std_rotate_copy);
    bm.operator()<std::vector<int>>("rng::rotate_copy(vector<int>)", std::ranges::rotate_copy);
    bm.operator()<std::deque<int>>("rng::rotate_copy(deque<int>)", std::ranges::rotate_copy);
    bm.operator()<std::list<int>>("rng::rotate_copy(list<int>)", std::ranges::rotate_copy);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
