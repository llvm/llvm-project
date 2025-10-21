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
#include <forward_list>
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

  auto register_benchmarks = [&](auto bm, std::string comment) {
    // {std,ranges}::find_end(it1, it1, it2, it2)
    bm.template operator()<std::vector<int>>("std::find_end(vector<int>) (" + comment + ")", std_find_end);
    bm.template operator()<std::deque<int>>("std::find_end(deque<int>) (" + comment + ")", std_find_end);
    bm.template operator()<std::list<int>>("std::find_end(list<int>) (" + comment + ")", std_find_end);
    bm.template operator()<std::forward_list<int>>("std::find_end(forward_list<int>) (" + comment + ")", std_find_end);
    bm.template operator()<std::vector<int>>("rng::find_end(vector<int>) (" + comment + ")", std::ranges::find_end);
    bm.template operator()<std::deque<int>>("rng::find_end(deque<int>) (" + comment + ")", std::ranges::find_end);
    bm.template operator()<std::list<int>>("rng::find_end(list<int>) (" + comment + ")", std::ranges::find_end);
    bm.template operator()<std::forward_list<int>>(
        "rng::find_end(forward_list<int>) (" + comment + ")", std::ranges::find_end);

    // {std,ranges}::find_end(it1, it1, it2, it2, pred)
    bm.template operator()<std::vector<int>>("std::find_end(vector<int>, pred) (" + comment + ")", std_find_end_pred);
    bm.template operator()<std::deque<int>>("std::find_end(deque<int>, pred) (" + comment + ")", std_find_end_pred);
    bm.template operator()<std::list<int>>("std::find_end(list<int>, pred) (" + comment + ")", std_find_end_pred);
    bm.template operator()<std::forward_list<int>>(
        "std::find_end(forward_list<int>, pred) (" + comment + ")", std_find_end_pred);
    bm.template operator()<std::vector<int>>(
        "rng::find_end(vector<int>, pred) (" + comment + ")", ranges_find_end_pred);
    bm.template operator()<std::deque<int>>("rng::find_end(deque<int>, pred) (" + comment + ")", ranges_find_end_pred);
    bm.template operator()<std::list<int>>("rng::find_end(list<int>, pred) (" + comment + ")", ranges_find_end_pred);
    bm.template operator()<std::forward_list<int>>(
        "rng::find_end(forward_list<int>, pred) (" + comment + ")", ranges_find_end_pred);
  };

  // Benchmark {std,ranges}::find_end where we never find the needle, which is the
  // worst case.
  {
    auto bm = []<class Container>(std::string name, auto find_end) {
      benchmark::RegisterBenchmark(
          name,
          [find_end](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            std::size_t n = size / 10; // needle size is 10% of the haystack, but we'll never find it
            assert(n > 0);
            Container needle(n, y);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = find_end(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    register_benchmarks(bm, "process all");
  }

  // Benchmark {std,ranges}::find_end where we intersperse "near matches" inside the haystack.
  {
    auto bm = []<class Container>(std::string name, auto find_end) {
      benchmark::RegisterBenchmark(
          name,
          [find_end](auto& st) {
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
              auto result = find_end(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    register_benchmarks(bm, "near matches");
  }

  // Special case: the two ranges are the same length (and they are equal, which is the worst case).
  {
    auto bm = []<class Container>(std::string name, auto find_end) {
      benchmark::RegisterBenchmark(
          name,
          [find_end](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            Container haystack(size, x);
            Container needle(size, x);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = find_end(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    register_benchmarks(bm, "same length");
  }

  // Special case: the needle contains a single element (which we never find, i.e. the worst case).
  {
    auto bm = []<class Container>(std::string name, auto find_end) {
      benchmark::RegisterBenchmark(
          name,
          [find_end](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            Container needle(1, y);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = find_end(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    register_benchmarks(bm, "single element");
  }

  // Special case: we have a match close to the end of the haystack (ideal case if we start searching from the end).
  {
    auto bm = []<class Container>(std::string name, auto find_end) {
      benchmark::RegisterBenchmark(
          name,
          [find_end](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            std::size_t n = size / 10; // needle size is 10% of the haystack
            assert(n > 0);
            Container needle(n, y);

            // put the needle at 90% of the haystack
            std::ranges::copy(needle, std::next(haystack.begin(), (9 * size) / 10));

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = find_end(haystack.begin(), haystack.end(), needle.begin(), needle.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    register_benchmarks(bm, "match near end");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
