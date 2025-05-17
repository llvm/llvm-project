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
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_shift_right = [](auto first, auto last, auto n) { return std::shift_right(first, last, n); };

  // Benchmark std::shift_right where we shift exactly one element, which is the worst case.
  {
    auto bm = []<class Container>(std::string name, auto shift_right) {
      benchmark::RegisterBenchmark(
          name,
          [shift_right](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });
            std::size_t n = 1;

            // To avoid ending up with a fully moved-from range, restore the element that gets
            // overwritten by the shift after performing the shift.
            auto first_element = c.begin();
            auto last_element  = std::next(c.begin(), size - 1);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(n);
              ValueType tmp  = *last_element;
              auto result    = shift_right(c.begin(), c.end(), n);
              *first_element = std::move(tmp);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::shift_right(vector<int>)", std_shift_right);
    bm.operator()<std::deque<int>>("std::shift_right(deque<int>)", std_shift_right);
    bm.operator()<std::list<int>>("std::shift_right(list<int>)", std_shift_right);
    // ranges::shift_right not implemented yet
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
