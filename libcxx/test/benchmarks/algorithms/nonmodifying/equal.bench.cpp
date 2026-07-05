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
  auto std_equal_3leg = [](auto first1, auto last1, auto first2, auto) { return std::equal(first1, last1, first2); };
  auto std_equal_4leg = [](auto first1, auto last1, auto first2, auto last2) {
    return std::equal(first1, last1, first2, last2);
  };
  auto std_equal_3leg_pred = [](auto first1, auto last1, auto first2, auto) {
    return std::equal(first1, last1, first2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };
  auto std_equal_4leg_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::equal(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  // Benchmark {std,ranges}::equal where we determine inequality at the very end (worst case).
  {
    auto bm = []<class Container>(std::string name, auto equal) {
      benchmark::RegisterBenchmark(
          name,
          [equal](auto& st) {
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
              auto result = equal(c1.begin(), c1.end(), c2.begin(), c2.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };

    // std::equal(it, it, it)
    bm.operator()<std::vector<int>>("std::equal(vector<int>) (it, it, it)", std_equal_3leg);
    bm.operator()<std::deque<int>>("std::equal(deque<int>) (it, it, it)", std_equal_3leg);
    bm.operator()<std::list<int>>("std::equal(list<int>) (it, it, it)", std_equal_3leg);

    // std::equal(it, it, it, pred)
    bm.operator()<std::vector<int>>("std::equal(vector<int>) (it, it, it, pred)", std_equal_3leg_pred);
    bm.operator()<std::deque<int>>("std::equal(deque<int>) (it, it, it, pred)", std_equal_3leg_pred);
    bm.operator()<std::list<int>>("std::equal(list<int>) (it, it, it, pred)", std_equal_3leg_pred);

    // {std,ranges}::equal(it, it, it, it)
    bm.operator()<std::vector<int>>("std::equal(vector<int>) (it, it, it, it)", std_equal_4leg);
    bm.operator()<std::deque<int>>("std::equal(deque<int>) (it, it, it, it)", std_equal_4leg);
    bm.operator()<std::list<int>>("std::equal(list<int>) (it, it, it, it)", std_equal_4leg);

    // {std,ranges}::equal(it, it, it, it, pred)
    bm.operator()<std::vector<int>>("std::equal(vector<int>) (it, it, it, it, pred)", std_equal_4leg_pred);
    bm.operator()<std::deque<int>>("std::equal(deque<int>) (it, it, it, it, pred)", std_equal_4leg_pred);
    bm.operator()<std::list<int>>("std::equal(list<int>) (it, it, it, it, pred)", std_equal_4leg_pred);
  }

  // Benchmark {std,ranges}::equal on vector<bool>.
  {
    auto bm = [](std::string name, auto equal, bool aligned) {
      benchmark::RegisterBenchmark(
          name,
          [=](auto& st) {
            std::size_t const size = st.range();
            std::vector<bool> c1(size, true);
            std::vector<bool> c2(size + 8, true);
            auto first1 = c1.begin();
            auto last1  = c1.end();
            auto first2 = aligned ? c2.begin() : c2.begin() + 4;
            auto last2  = aligned ? c2.end() : c2.end() - 4;
            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c1);
              benchmark::DoNotOptimize(c2);
              auto result = equal(first1, last1, first2, last2);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };

    // {std,ranges}::equal(vector<bool>) (aligned)
    bm("std::equal(vector<bool>) (aligned)", std_equal_4leg, true);

    // {std,ranges}::equal(vector<bool>) (unaligned)
    bm("std::equal(vector<bool>) (unaligned)", std_equal_4leg, false);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
