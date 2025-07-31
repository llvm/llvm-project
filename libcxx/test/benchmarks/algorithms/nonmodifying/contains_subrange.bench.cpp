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
#include <iterator>
#include <list>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  // Benchmark ranges::contains_subrange where we never find the needle, which is the
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
            Container haystack(size, x);
            std::size_t n = size / 10; // needle size is 10% of the haystack, but we'll never find it
            assert(n > 0);
            Container needle(n, y);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = std::ranges::contains_subrange(haystack, needle);
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

  // Benchmark ranges::contains_subrange where we intersperse "near matches" inside the haystack.
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            std::size_t n = size / 10; // needle size is 10% of the haystack, but we'll never find it
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
              auto result = std::ranges::contains_subrange(haystack, needle);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("rng::contains_subrange(vector<int>) (near matches)");
    bm.operator()<std::deque<int>>("rng::contains_subrange(deque<int>) (near matches)");
    bm.operator()<std::list<int>>("rng::contains_subrange(list<int>) (near matches)");
  }

  // Special case: the two ranges are the same length (and they are equal, which is the worst case).
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            Container haystack(size, x);
            Container needle(size, x);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = std::ranges::contains_subrange(haystack, needle);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(16)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::contains_subrange(vector<int>) (same length)");
    bm.operator()<std::deque<int>>("rng::contains_subrange(deque<int>) (same length)");
    bm.operator()<std::list<int>>("rng::contains_subrange(list<int>) (same length)");
  }

  // Special case: the needle contains a single element (which we never find, i.e. the worst case).
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            Container needle(1, y);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(needle);
              auto result = std::ranges::contains_subrange(haystack, needle);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(16)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::contains_subrange(vector<int>) (single element)");
    bm.operator()<std::deque<int>>("rng::contains_subrange(deque<int>) (single element)");
    bm.operator()<std::list<int>>("rng::contains_subrange(list<int>) (single element)");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
