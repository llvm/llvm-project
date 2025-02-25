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
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_find_end = [](auto first1, auto last1, auto first2, auto last2) {
    return std::find_end(first1, last1, first2, last2);
  };
  auto std_find_end_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::find_end(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };
  auto ranges_find_end_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::ranges::find_end(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark {std,ranges}::find_end where the subsequence is found
  // 25% into the sequence
  {
    auto bm = []<class Container>(std::string name, auto find_end) {
      benchmark::RegisterBenchmark(
          name,
          [find_end](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);
            Container subrange(size / 10, y); // subrange of length 10% of the full range

            // put the element we're searching for at 25% of the sequence
            std::ranges::copy(subrange, std::next(c.begin(), c.size() / 4));

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(subrange);
              auto result = find_end(c.begin(), c.end(), subrange.begin(), subrange.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    // {std,ranges}::find_end(it1, it1, it2, it2)
    bm.operator()<std::vector<int>>("std::find_end(vector<int>) (bail 25%)", std_find_end);
    bm.operator()<std::deque<int>>("std::find_end(deque<int>) (bail 25%)", std_find_end);
    bm.operator()<std::list<int>>("std::find_end(list<int>) (bail 25%)", std_find_end);
    bm.operator()<std::vector<int>>("rng::find_end(vector<int>) (bail 25%)", std::ranges::find_end);
    bm.operator()<std::deque<int>>("rng::find_end(deque<int>) (bail 25%)", std::ranges::find_end);
    bm.operator()<std::list<int>>("rng::find_end(list<int>) (bail 25%)", std::ranges::find_end);

    // {std,ranges}::find_end(it1, it1, it2, it2, pred)
    bm.operator()<std::vector<int>>("std::find_end(vector<int>, pred) (bail 25%)", std_find_end_pred);
    bm.operator()<std::deque<int>>("std::find_end(deque<int>, pred) (bail 25%)", std_find_end_pred);
    bm.operator()<std::list<int>>("std::find_end(list<int>, pred) (bail 25%)", std_find_end_pred);
    bm.operator()<std::vector<int>>("rng::find_end(vector<int>, pred) (bail 25%)", ranges_find_end_pred);
    bm.operator()<std::deque<int>>("rng::find_end(deque<int>, pred) (bail 25%)", ranges_find_end_pred);
    bm.operator()<std::list<int>>("rng::find_end(list<int>, pred) (bail 25%)", ranges_find_end_pred);
  }

  // Benchmark {std,ranges}::find_end where the subsequence is found
  // 90% into the sequence (i.e. near the end)
  {
    auto bm = []<class Container>(std::string name, auto find_end) {
      benchmark::RegisterBenchmark(
          name,
          [find_end](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);
            Container subrange(size / 10, y); // subrange of length 10% of the full range

            // put the element we're searching for at 90% of the sequence
            std::ranges::copy(subrange, std::next(c.begin(), 9 * (c.size() / 10)));

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(subrange);
              auto result = find_end(c.begin(), c.end(), subrange.begin(), subrange.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    // {std,ranges}::find_end(it1, it1, it2, it2)
    bm.operator()<std::vector<int>>("std::find_end(vector<int>) (bail 90%)", std_find_end);
    bm.operator()<std::deque<int>>("std::find_end(deque<int>) (bail 90%)", std_find_end);
    bm.operator()<std::list<int>>("std::find_end(list<int>) (bail 90%)", std_find_end);
    bm.operator()<std::vector<int>>("rng::find_end(vector<int>) (bail 90%)", std::ranges::find_end);
    bm.operator()<std::deque<int>>("rng::find_end(deque<int>) (bail 90%)", std::ranges::find_end);
    bm.operator()<std::list<int>>("rng::find_end(list<int>) (bail 90%)", std::ranges::find_end);

    // {std,ranges}::find_end(it1, it1, it2, it2, pred)
    bm.operator()<std::vector<int>>("std::find_end(vector<int>, pred) (bail 90%)", std_find_end_pred);
    bm.operator()<std::deque<int>>("std::find_end(deque<int>, pred) (bail 90%)", std_find_end_pred);
    bm.operator()<std::list<int>>("std::find_end(list<int>, pred) (bail 90%)", std_find_end_pred);
    bm.operator()<std::vector<int>>("rng::find_end(vector<int>, pred) (bail 90%)", ranges_find_end_pred);
    bm.operator()<std::deque<int>>("rng::find_end(deque<int>, pred) (bail 90%)", ranges_find_end_pred);
    bm.operator()<std::list<int>>("rng::find_end(list<int>, pred) (bail 90%)", ranges_find_end_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
