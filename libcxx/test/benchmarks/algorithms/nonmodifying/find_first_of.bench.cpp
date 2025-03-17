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
  auto ranges_find_first_of_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::ranges::find_first_of(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark {std,ranges}::find_first_of where we have a hit at 10% of the haystack
  // and at the end of the needle. This measures how quickly we're able to search inside
  // the needle.
  {
    auto bm = []<class Container>(std::string name, auto find_first_of) {
      benchmark::RegisterBenchmark(
          name,
          [find_first_of](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            ValueType z            = random_different_from({x, y});
            Container haystack(size, x);
            Container needle(size, y);
            needle.back() = z; // hit at the very end of the needle

            // put the needle at 10% of the haystack
            *std::next(haystack.begin(), haystack.size() / 10) = z;

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
    bm.operator()<std::vector<int>>("std::find_first_of(vector<int>) (10% haystack, late needle)", std_find_first_of);
    bm.operator()<std::deque<int>>("std::find_first_of(deque<int>) (10% haystack, late needle)", std_find_first_of);
    bm.operator()<std::list<int>>("std::find_first_of(list<int>) (10% haystack, late needle)", std_find_first_of);
    bm.operator()<std::vector<int>>(
        "rng::find_first_of(vector<int>) (10% haystack, late needle)", std::ranges::find_first_of);
    bm.operator()<std::deque<int>>(
        "rng::find_first_of(deque<int>) (10% haystack, late needle)", std::ranges::find_first_of);
    bm.operator()<std::list<int>>(
        "rng::find_first_of(list<int>) (10% haystack, late needle)", std::ranges::find_first_of);

    // {std,ranges}::find_first_of(it1, it1, it2, it2, pred)
    bm.operator()<std::vector<int>>(
        "std::find_first_of(vector<int>, pred) (25% haystack, late needle)", std_find_first_of);
    bm.operator()<std::deque<int>>(
        "std::find_first_of(deque<int>, pred) (25% haystack, late needle)", std_find_first_of);
    bm.operator()<std::list<int>>("std::find_first_of(list<int>, pred) (25% haystack, late needle)", std_find_first_of);
    bm.operator()<std::vector<int>>(
        "rng::find_first_of(vector<int>, pred) (25% haystack, late needle)", std::ranges::find_first_of);
    bm.operator()<std::deque<int>>(
        "rng::find_first_of(deque<int>, pred) (25% haystack, late needle)", std::ranges::find_first_of);
    bm.operator()<std::list<int>>(
        "rng::find_first_of(list<int>, pred) (25% haystack, late needle)", std::ranges::find_first_of);
  }

  // Benchmark {std,ranges}::find_first_of where we have a hit at 90% of the haystack
  // but at the beginning of the needle. This measures how quickly we're able to search
  // inside the haystack.
  {
    auto bm = []<class Container>(std::string name, auto find_first_of) {
      benchmark::RegisterBenchmark(
          name,
          [find_first_of](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            ValueType z            = random_different_from({x, y});
            Container haystack(size, x);
            Container needle(size, y);
            *std::next(needle.begin(), needle.size() / 10) = z; // hit at 10% of the needle

            // put the needle at 90% of the haystack
            *std::next(haystack.begin(), (9 * haystack.size()) / 10) = z;

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
    bm.operator()<std::vector<int>>("std::find_first_of(vector<int>) (90% haystack, early needle)", std_find_first_of);
    bm.operator()<std::deque<int>>("std::find_first_of(deque<int>) (90% haystack, early needle)", std_find_first_of);
    bm.operator()<std::list<int>>("std::find_first_of(list<int>) (90% haystack, early needle)", std_find_first_of);
    bm.operator()<std::vector<int>>(
        "rng::find_first_of(vector<int>) (90% haystack, early needle)", std::ranges::find_first_of);
    bm.operator()<std::deque<int>>(
        "rng::find_first_of(deque<int>) (90% haystack, early needle)", std::ranges::find_first_of);
    bm.operator()<std::list<int>>(
        "rng::find_first_of(list<int>) (90% haystack, early needle)", std::ranges::find_first_of);

    // {std,ranges}::find_first_of(it1, it1, it2, it2, pred)
    bm.operator()<std::vector<int>>(
        "std::find_first_of(vector<int>, pred) (90% haystack, early needle)", std_find_first_of_pred);
    bm.operator()<std::deque<int>>(
        "std::find_first_of(deque<int>, pred) (90% haystack, early needle)", std_find_first_of_pred);
    bm.operator()<std::list<int>>(
        "std::find_first_of(list<int>, pred) (90% haystack, early needle)", std_find_first_of_pred);
    bm.operator()<std::vector<int>>(
        "rng::find_first_of(vector<int>, pred) (90% haystack, early needle)", ranges_find_first_of_pred);
    bm.operator()<std::deque<int>>(
        "rng::find_first_of(deque<int>, pred) (90% haystack, early needle)", ranges_find_first_of_pred);
    bm.operator()<std::list<int>>(
        "rng::find_first_of(list<int>, pred) (90% haystack, early needle)", ranges_find_first_of_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
