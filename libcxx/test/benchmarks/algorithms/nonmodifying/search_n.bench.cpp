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
  auto std_search_n = [](auto first, auto last, auto n, auto const& value) {
    return std::search_n(first, last, n, value);
  };
  auto std_search_n_pred = [](auto first, auto last, auto n, auto const& value) {
    return std::search_n(first, last, n, value, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };
  auto ranges_search_n_pred = [](auto first, auto last, auto n, auto const& value) {
    return std::ranges::search_n(first, last, n, value, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark {std,ranges}::search_n where the needle is never found (worst case).
  {
    auto bm = []<class Container>(std::string name, auto search_n) {
      benchmark::RegisterBenchmark(
          name,
          [search_n](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container haystack(size, x);
            std::size_t n = size / 10; // needle size is 10% of the haystack

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(haystack);
              benchmark::DoNotOptimize(n);
              benchmark::DoNotOptimize(y);
              auto result = search_n(haystack.begin(), haystack.end(), n, y);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    // {std,ranges}::search_n
    bm.operator()<std::vector<int>>("std::search_n(vector<int>) (no match)", std_search_n);
    bm.operator()<std::deque<int>>("std::search_n(deque<int>) (no match)", std_search_n);
    bm.operator()<std::list<int>>("std::search_n(list<int>) (no match)", std_search_n);
    bm.operator()<std::vector<int>>("rng::search_n(vector<int>) (no match)", std::ranges::search_n);
    bm.operator()<std::deque<int>>("rng::search_n(deque<int>) (no match)", std::ranges::search_n);
    bm.operator()<std::list<int>>("rng::search_n(list<int>) (no match)", std::ranges::search_n);

    // {std,ranges}::search_n(pred)
    bm.operator()<std::vector<int>>("std::search_n(vector<int>, pred) (no match)", std_search_n_pred);
    bm.operator()<std::deque<int>>("std::search_n(deque<int>, pred) (no match)", std_search_n_pred);
    bm.operator()<std::list<int>>("std::search_n(list<int>, pred) (no match)", std_search_n_pred);
    bm.operator()<std::vector<int>>("rng::search_n(vector<int>, pred) (no match)", ranges_search_n_pred);
    bm.operator()<std::deque<int>>("rng::search_n(deque<int>, pred) (no match)", ranges_search_n_pred);
    bm.operator()<std::list<int>>("rng::search_n(list<int>, pred) (no match)", ranges_search_n_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
