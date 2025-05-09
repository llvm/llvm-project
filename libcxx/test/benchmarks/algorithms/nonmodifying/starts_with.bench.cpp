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
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto ranges_starts_with_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::ranges::starts_with(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark ranges::starts_with where we find the mismatching element at the very end (worst case).
  {
    auto bm = []<class Container>(std::string name, auto starts_with) {
      benchmark::RegisterBenchmark(
          name,
          [starts_with](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c1(size, x);
            Container c2(size, x);
            c2.back() = y;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c1);
              benchmark::DoNotOptimize(c2);
              auto result = starts_with(c1.begin(), c1.end(), c2.begin(), c2.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::starts_with(vector<int>)", std::ranges::starts_with);
    bm.operator()<std::deque<int>>("rng::starts_with(deque<int>)", std::ranges::starts_with);
    bm.operator()<std::list<int>>("rng::starts_with(list<int>)", std::ranges::starts_with);

    bm.operator()<std::vector<int>>("rng::starts_with(vector<int>, pred)", ranges_starts_with_pred);
    bm.operator()<std::deque<int>>("rng::starts_with(deque<int>, pred)", ranges_starts_with_pred);
    bm.operator()<std::list<int>>("rng::starts_with(list<int>, pred)", ranges_starts_with_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
