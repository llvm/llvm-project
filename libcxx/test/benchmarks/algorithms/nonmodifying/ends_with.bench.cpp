//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <forward_list>
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto ranges_ends_with_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::ranges::ends_with(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark ranges::ends_with where we find the mismatching element at the very end.
  {
    auto bm = []<class Container>(std::string name, auto ends_with) {
      benchmark::RegisterBenchmark(
          name,
          [ends_with](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c1(size, x);
            Container c2(size, x);
            assert(size != 0);
            *std::next(c2.begin(), size - 1) = y; // set last element to y

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c1);
              benchmark::DoNotOptimize(c2);
              auto result = ends_with(c1.begin(), c1.end(), c2.begin(), c2.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::ends_with(vector<int>) (mismatch at end)", std::ranges::ends_with);
    bm.operator()<std::deque<int>>("rng::ends_with(deque<int>) (mismatch at end)", std::ranges::ends_with);
    bm.operator()<std::list<int>>("rng::ends_with(list<int>) (mismatch at end)", std::ranges::ends_with);
    bm.operator()<std::forward_list<int>>(
        "rng::ends_with(forward_list<int>) (mismatch at end)", std::ranges::ends_with);

    bm.operator()<std::vector<int>>("rng::ends_with(vector<int>, pred) (mismatch at end)", ranges_ends_with_pred);
    bm.operator()<std::deque<int>>("rng::ends_with(deque<int>, pred) (mismatch at end)", ranges_ends_with_pred);
    bm.operator()<std::list<int>>("rng::ends_with(list<int>, pred) (mismatch at end)", ranges_ends_with_pred);
    bm.operator()<std::forward_list<int>>(
        "rng::ends_with(forward_list<int>, pred) (mismatch at end)", ranges_ends_with_pred);
  }

  // Benchmark ranges::ends_with where we find the mismatching element at the very beginning.
  {
    auto bm = []<class Container>(std::string name, auto ends_with) {
      benchmark::RegisterBenchmark(
          name,
          [ends_with](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c1(size, x);
            Container c2(size, x);
            assert(size != 0);
            c2.front() = y;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c1);
              benchmark::DoNotOptimize(c2);
              auto result = ends_with(c1.begin(), c1.end(), c2.begin(), c2.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::ends_with(vector<int>) (mismatch at start)", std::ranges::ends_with);
    bm.operator()<std::deque<int>>("rng::ends_with(deque<int>) (mismatch at start)", std::ranges::ends_with);
    bm.operator()<std::list<int>>("rng::ends_with(list<int>) (mismatch at start)", std::ranges::ends_with);
    bm.operator()<std::forward_list<int>>(
        "rng::ends_with(forward_list<int>) (mismatch at start)", std::ranges::ends_with);

    bm.operator()<std::vector<int>>("rng::ends_with(vector<int>, pred) (mismatch at start)", ranges_ends_with_pred);
    bm.operator()<std::deque<int>>("rng::ends_with(deque<int>, pred) (mismatch at start)", ranges_ends_with_pred);
    bm.operator()<std::list<int>>("rng::ends_with(list<int>, pred) (mismatch at start)", ranges_ends_with_pred);
    bm.operator()<std::forward_list<int>>(
        "rng::ends_with(forward_list<int>, pred) (mismatch at start)", ranges_ends_with_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
