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
  // Benchmark ranges::contains_subrange where we never find our target, which is the
  // worst case.
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
