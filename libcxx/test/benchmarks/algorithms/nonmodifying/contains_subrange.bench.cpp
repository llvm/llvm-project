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
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  // Benchmark ranges::contains_subrange where we find our target starting at 25% of the elements
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);
            Container subrange(size / 10, y); // subrange of length 10% of the full range

            // At 25% of the range, put the subrange we're going to find
            std::ranges::copy(subrange, std::next(c.begin(), c.size() / 4));

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(subrange);
              auto result = std::ranges::contains_subrange(c, subrange);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(16)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::contains_subrange(vector<int>) (bail 25%)");
    bm.operator()<std::deque<int>>("rng::contains_subrange(deque<int>) (bail 25%)");
    bm.operator()<std::list<int>>("rng::contains_subrange(list<int>) (bail 25%)");
  }

  // Benchmark ranges::contains_subrange where we never find our target
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);
            Container subrange(size / 10, y); // subrange of length 10% of the full range, but we'll never find it

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(subrange);
              auto result = std::ranges::contains_subrange(c, subrange);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(16)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::contains_subrange(vector<int>) (process all)");
    bm.operator()<std::deque<int>>("rng::contains_subrange(deque<int>) (process all)");
    bm.operator()<std::list<int>>("rng::contains_subrange(list<int>) (process all)");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
