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
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  auto std_for_each_n = [](auto first, auto n, auto f) { return std::for_each_n(first, n, f); };

  // {std,ranges}::for_each_n
  {
    auto bm = []<class Container>(std::string name, auto for_each_n) {
      benchmark::RegisterBenchmark(
          name,
          [for_each_n](auto& st) {
            std::size_t const n = st.range(0);
            Container c(n, 1);
            auto first = c.begin();

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = for_each_n(first, n, [](int& x) { x = std::clamp(x, 10, 100); });
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("std::for_each_n(vector<int>)", std_for_each_n);
    bm.operator()<std::deque<int>>("std::for_each_n(deque<int>)", std_for_each_n);
    bm.operator()<std::list<int>>("std::for_each_n(list<int>)", std_for_each_n);
    bm.operator()<std::vector<int>>("rng::for_each_n(vector<int>)", std::ranges::for_each_n);
    bm.operator()<std::deque<int>>("rng::for_each_n(deque<int>)", std::ranges::for_each_n);
    bm.operator()<std::list<int>>("rng::for_each_n(list<int>)", std::ranges::for_each_n);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
