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
#include <ranges>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"
#include "test_macros.h"

int main(int argc, char** argv) {
  auto std_rotate = [](auto first, auto middle, auto last) { return std::rotate(first, middle, last); };

  // Benchmark {std,ranges}::rotate where we rotate various fractions of the range. It is possible to
  // special-case some of these fractions to cleverly perform swap_ranges.
  {
    auto bm = []<class Container>(std::string name, auto rotate, double fraction) {
      benchmark::RegisterBenchmark(
          name,
          [=](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

            auto nth = std::next(c.begin(), static_cast<std::size_t>(size * fraction));
            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = rotate(c.begin(), nth, c.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->RangeMultiplier(2)
          ->Range(64, 1 << 20);
    };

    bm.operator()<std::vector<bool>>("std::rotate(vector<bool>) (by 1/4)", std_rotate, 0.25);
    bm.operator()<std::vector<bool>>("std::rotate(vector<bool>) (by 51%)", std_rotate, 0.51);
#if TEST_STD_VER >= 23 // vector<bool>::iterator is not std::permutable before C++23
    bm.operator()<std::vector<bool>>("rng::rotate(vector<bool>) (by 1/4)", std::ranges::rotate, 0.25);
    bm.operator()<std::vector<bool>>("rng::rotate(vector<bool>) (by 51%)", std::ranges::rotate, 0.51);
#endif
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
