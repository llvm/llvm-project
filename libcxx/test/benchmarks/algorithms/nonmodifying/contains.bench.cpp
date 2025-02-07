//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <list>
#include <vector>

#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  // Benchmark ranges::contains where we bail out early (after visiting 25% of the elements).
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) {
            std::size_t const size = st.range(0);
            Container c(size, 1);
            *std::next(c.begin(), size / 4) = 42; // bail out after checking 25% of values
            auto first                      = c.begin();
            auto last                       = c.end();

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = std::ranges::contains(first, last, 42);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::contains(vector<int>) (bail 25%)");
    bm.operator()<std::deque<int>>("rng::contains(deque<int>) (bail 25%)");
    bm.operator()<std::list<int>>("rng::contains(list<int>) (bail 25%)");
  }

  // Benchmark ranges::contains where we process the whole sequence.
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) {
            std::size_t const size = st.range(0);
            Container c(size, 1);
            auto first = c.begin();
            auto last  = c.end();

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = std::ranges::contains(first, last, 42);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::contains(vector<int>) (process all)");
    bm.operator()<std::deque<int>>("rng::contains(deque<int>) (process all)");
    bm.operator()<std::list<int>>("rng::contains(list<int>) (process all)");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
