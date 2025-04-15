//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_search = [](auto first1, auto last1, auto first2, auto last2) {
    return std::search(first1, last1, first2, last2);
  };
  auto std_search_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::search(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };
  auto ranges_search_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::ranges::search(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark {std,ranges}::search where the needle is never found (worst case).
  {
    auto bm = []<class Container>(std::string name, auto search) {
      benchmark::RegisterBenchmark(
          name,
          [search](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            Container needle(size / 10, y); // needle size is 10% of the haystack

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    // {std,ranges}::search
    bm.operator()<std::vector<int>>("std::search(vector<int>) (no match)", std_search);
    bm.operator()<std::deque<int>>("std::search(deque<int>) (no match)", std_search);
    bm.operator()<std::list<int>>("std::search(list<int>) (no match)", std_search);
    bm.operator()<std::vector<int>>("rng::search(vector<int>) (no match)", std::ranges::search);
    bm.operator()<std::deque<int>>("rng::search(deque<int>) (no match)", std::ranges::search);
    bm.operator()<std::list<int>>("rng::search(list<int>) (no match)", std::ranges::search);

    // {std,ranges}::search(pred)
    bm.operator()<std::vector<int>>("std::search(vector<int>, pred) (no match)", std_search_pred);
    bm.operator()<std::deque<int>>("std::search(deque<int>, pred) (no match)", std_search_pred);
    bm.operator()<std::list<int>>("std::search(list<int>, pred) (no match)", std_search_pred);
    bm.operator()<std::vector<int>>("rng::search(vector<int>, pred) (no match)", ranges_search_pred);
    bm.operator()<std::deque<int>>("rng::search(deque<int>, pred) (no match)", ranges_search_pred);
    bm.operator()<std::list<int>>("rng::search(list<int>, pred) (no match)", ranges_search_pred);
  }

  // Benchmark {std,ranges}::search where we intersperse "near matches" inside the haystack.
  {
    auto bm = []<class Container>(std::string name, auto search) {
      benchmark::RegisterBenchmark(
          name,
          [search](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            std::size_t n = size / 10; // needle size is 10% of the haystack
            assert(n > 0);
            Container needle(n, y);

            // intersperse near-matches inside the haystack
            {
              auto first = haystack.begin();
              for (int i = 0; i != 10; ++i) {
                first = std::copy_n(needle.begin(), n - 1, first);
                ++first; // this causes the subsequence not to match because it has length n-1
              }
            }

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::search
    bm.operator()<std::vector<int>>("std::search(vector<int>) (near matches)", std_search);
    bm.operator()<std::deque<int>>("std::search(deque<int>) (near matches)", std_search);
    bm.operator()<std::list<int>>("std::search(list<int>) (near matches)", std_search);
    bm.operator()<std::vector<int>>("rng::search(vector<int>) (near matches)", std::ranges::search);
    bm.operator()<std::deque<int>>("rng::search(deque<int>) (near matches)", std::ranges::search);
    bm.operator()<std::list<int>>("rng::search(list<int>) (near matches)", std::ranges::search);

    // {std,ranges}::search(pred)
    bm.operator()<std::vector<int>>("std::search(vector<int>, pred) (near matches)", std_search_pred);
    bm.operator()<std::deque<int>>("std::search(deque<int>, pred) (near matches)", std_search_pred);
    bm.operator()<std::list<int>>("std::search(list<int>, pred) (near matches)", std_search_pred);
    bm.operator()<std::vector<int>>("rng::search(vector<int>, pred) (near matches)", ranges_search_pred);
    bm.operator()<std::deque<int>>("rng::search(deque<int>, pred) (near matches)", ranges_search_pred);
    bm.operator()<std::list<int>>("rng::search(list<int>, pred) (near matches)", ranges_search_pred);
  }

  // Special case: the two ranges are the same length (and they are equal, which is the worst case).
  {
    auto bm = []<class Container>(std::string name, auto search) {
      benchmark::RegisterBenchmark(
          name,
          [search](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            Container haystack(size, x);
            Container needle(size, x);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::search
    bm.operator()<std::vector<int>>("std::search(vector<int>) (same length)", std_search);
    bm.operator()<std::deque<int>>("std::search(deque<int>) (same length)", std_search);
    bm.operator()<std::list<int>>("std::search(list<int>) (same length)", std_search);
    bm.operator()<std::vector<int>>("rng::search(vector<int>) (same length)", std::ranges::search);
    bm.operator()<std::deque<int>>("rng::search(deque<int>) (same length)", std::ranges::search);
    bm.operator()<std::list<int>>("rng::search(list<int>) (same length)", std::ranges::search);

    // {std,ranges}::search(pred)
    bm.operator()<std::vector<int>>("std::search(vector<int>, pred) (same length)", std_search_pred);
    bm.operator()<std::deque<int>>("std::search(deque<int>, pred) (same length)", std_search_pred);
    bm.operator()<std::list<int>>("std::search(list<int>, pred) (same length)", std_search_pred);
    bm.operator()<std::vector<int>>("rng::search(vector<int>, pred) (same length)", ranges_search_pred);
    bm.operator()<std::deque<int>>("rng::search(deque<int>, pred) (same length)", ranges_search_pred);
    bm.operator()<std::list<int>>("rng::search(list<int>, pred) (same length)", ranges_search_pred);
  }

  // Special case: the needle contains a single element (which we never find, i.e. the worst case).
  {
    auto bm = []<class Container>(std::string name, auto search) {
      benchmark::RegisterBenchmark(
          name,
          [search](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            Container needle(1, y);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::search
    bm.operator()<std::vector<int>>("std::search(vector<int>) (single element)", std_search);
    bm.operator()<std::deque<int>>("std::search(deque<int>) (single element)", std_search);
    bm.operator()<std::list<int>>("std::search(list<int>) (single element)", std_search);
    bm.operator()<std::vector<int>>("rng::search(vector<int>) (single element)", std::ranges::search);
    bm.operator()<std::deque<int>>("rng::search(deque<int>) (single element)", std::ranges::search);
    bm.operator()<std::list<int>>("rng::search(list<int>) (single element)", std::ranges::search);

    // {std,ranges}::search(pred)
    bm.operator()<std::vector<int>>("std::search(vector<int>, pred) (single element)", std_search_pred);
    bm.operator()<std::deque<int>>("std::search(deque<int>, pred) (single element)", std_search_pred);
    bm.operator()<std::list<int>>("std::search(list<int>, pred) (single element)", std_search_pred);
    bm.operator()<std::vector<int>>("rng::search(vector<int>, pred) (single element)", ranges_search_pred);
    bm.operator()<std::deque<int>>("rng::search(deque<int>, pred) (single element)", ranges_search_pred);
    bm.operator()<std::list<int>>("rng::search(list<int>, pred) (single element)", ranges_search_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
