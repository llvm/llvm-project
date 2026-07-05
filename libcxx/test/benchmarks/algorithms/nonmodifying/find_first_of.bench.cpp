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
#include <initializer_list>
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_find_first_of = [](auto first1, auto last1, auto first2, auto last2) {
    return std::find_first_of(first1, last1, first2, last2);
  };
  auto std_find_first_of_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::find_first_of(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark {std,ranges}::find_first_of where we never find a match in the needle, and the needle is small.
  // This is the worst case of the most common case (a small needle).
  {
    auto bm = []<class Container>(std::string name, auto find_first_of) {
      benchmark::RegisterBenchmark(
          name,
          [find_first_of](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            Container needle(10, y);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = find_first_of(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::find_first_of(it1, it1, it2, it2)
    bm.operator()<std::vector<int>>("std::find_first_of(vector<int>) (small needle)", std_find_first_of);
    bm.operator()<std::deque<int>>("std::find_first_of(deque<int>) (small needle)", std_find_first_of);
    bm.operator()<std::list<int>>("std::find_first_of(list<int>) (small needle)", std_find_first_of);

    // {std,ranges}::find_first_of(it1, it1, it2, it2, pred)
    bm.operator()<std::vector<int>>("std::find_first_of(vector<int>, pred) (small needle)", std_find_first_of_pred);
    bm.operator()<std::deque<int>>("std::find_first_of(deque<int>, pred) (small needle)", std_find_first_of_pred);
    bm.operator()<std::list<int>>("std::find_first_of(list<int>, pred) (small needle)", std_find_first_of_pred);
  }

  // Special case: the needle is large compared to the haystack, and we find a match early in the haystack.
  {
    auto bm = []<class Container>(std::string name, auto find_first_of) {
      benchmark::RegisterBenchmark(
          name,
          [find_first_of](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            Container needle(size * 10, y);

            // put a match at 10% of the haystack
            *std::next(haystack.begin(), haystack.size() / 10) = y;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = find_first_of(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::find_first_of(it1, it1, it2, it2)
    bm.operator()<std::vector<int>>("std::find_first_of(vector<int>) (large needle)", std_find_first_of);
    bm.operator()<std::deque<int>>("std::find_first_of(deque<int>) (large needle)", std_find_first_of);
    bm.operator()<std::list<int>>("std::find_first_of(list<int>) (large needle)", std_find_first_of);

    // {std,ranges}::find_first_of(it1, it1, it2, it2, pred)
    bm.operator()<std::vector<int>>("std::find_first_of(vector<int>, pred) (large needle)", std_find_first_of_pred);
    bm.operator()<std::deque<int>>("std::find_first_of(deque<int>, pred) (large needle)", std_find_first_of_pred);
    bm.operator()<std::list<int>>("std::find_first_of(list<int>, pred) (large needle)", std_find_first_of_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
