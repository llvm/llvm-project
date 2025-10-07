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
  auto std_mismatch_3leg = [](auto first1, auto last1, auto first2, auto) {
    return std::mismatch(first1, last1, first2);
  };
  auto std_mismatch_4leg = [](auto first1, auto last1, auto first2, auto last2) {
    return std::mismatch(first1, last1, first2, last2);
  };
  auto std_mismatch_3leg_pred = [](auto first1, auto last1, auto first2, auto) {
    return std::mismatch(first1, last1, first2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };
  auto std_mismatch_4leg_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::mismatch(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };
  auto ranges_mismatch_4leg_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::ranges::mismatch(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark {std,ranges}::mismatch where we find the mismatching element at the very end (worst case).
  //
  // TODO: Look into benchmarking aligned and unaligned memory explicitly
  // (currently things happen to be aligned because they are malloced that way)
  {
    auto bm = []<class Container>(std::string name, auto mismatch) {
      benchmark::RegisterBenchmark(
          name,
          [mismatch](auto& st) {
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
              auto result = mismatch(c1.begin(), c1.end(), c2.begin(), c2.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1000) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };

    // std::mismatch(it, it, it)
    bm.operator()<std::vector<int>>("std::mismatch(vector<int>) (it, it, it)", std_mismatch_3leg);
    bm.operator()<std::deque<int>>("std::mismatch(deque<int>) (it, it, it)", std_mismatch_3leg);
    bm.operator()<std::list<int>>("std::mismatch(list<int>) (it, it, it)", std_mismatch_3leg);

    // std::mismatch(it, it, it, pred)
    bm.operator()<std::vector<int>>("std::mismatch(vector<int>) (it, it, it, pred)", std_mismatch_3leg_pred);
    bm.operator()<std::deque<int>>("std::mismatch(deque<int>) (it, it, it, pred)", std_mismatch_3leg_pred);
    bm.operator()<std::list<int>>("std::mismatch(list<int>) (it, it, it, pred)", std_mismatch_3leg_pred);

    // {std,ranges}::mismatch(it, it, it, it)
    bm.operator()<std::vector<int>>("std::mismatch(vector<int>) (it, it, it, it)", std_mismatch_4leg);
    bm.operator()<std::deque<int>>("std::mismatch(deque<int>) (it, it, it, it)", std_mismatch_4leg);
    bm.operator()<std::list<int>>("std::mismatch(list<int>) (it, it, it, it)", std_mismatch_4leg);
    bm.operator()<std::vector<int>>("rng::mismatch(vector<int>) (it, it, it, it)", std::ranges::mismatch);
    bm.operator()<std::deque<int>>("rng::mismatch(deque<int>) (it, it, it, it)", std::ranges::mismatch);
    bm.operator()<std::list<int>>("rng::mismatch(list<int>) (it, it, it, it)", std::ranges::mismatch);

    // {std,ranges}::mismatch(it, it, it, it, pred)
    bm.operator()<std::vector<int>>("std::mismatch(vector<int>) (it, it, it, it, pred)", std_mismatch_4leg_pred);
    bm.operator()<std::deque<int>>("std::mismatch(deque<int>) (it, it, it, it, pred)", std_mismatch_4leg_pred);
    bm.operator()<std::list<int>>("std::mismatch(list<int>) (it, it, it, it, pred)", std_mismatch_4leg_pred);
    bm.operator()<std::vector<int>>("rng::mismatch(vector<int>) (it, it, it, it, pred)", ranges_mismatch_4leg_pred);
    bm.operator()<std::deque<int>>("rng::mismatch(deque<int>) (it, it, it, it, pred)", ranges_mismatch_4leg_pred);
    bm.operator()<std::list<int>>("rng::mismatch(list<int>) (it, it, it, it, pred)", ranges_mismatch_4leg_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
