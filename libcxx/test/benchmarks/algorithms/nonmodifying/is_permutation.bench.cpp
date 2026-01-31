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
#include <iterator>
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_is_permutation_3leg = [](auto first1, auto last1, auto first2, auto) {
    return std::is_permutation(first1, last1, first2);
  };
  auto std_is_permutation_4leg = [](auto first1, auto last1, auto first2, auto last2) {
    return std::is_permutation(first1, last1, first2, last2);
  };
  auto std_is_permutation_3leg_pred = [](auto first1, auto last1, auto first2, auto) {
    return std::is_permutation(first1, last1, first2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };
  auto std_is_permutation_4leg_pred = [](auto first1, auto last1, auto first2, auto last2) {
    return std::is_permutation(first1, last1, first2, last2, [](auto x, auto y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x == y;
    });
  };

  auto register_benchmarks = [&](auto bm, std::string comment) {
    // std::is_permutation(it, it, it)
    bm.template operator()<std::vector<int>>(
        "std::is_permutation(vector<int>) (3leg) (" + comment + ")", std_is_permutation_3leg);
    bm.template operator()<std::deque<int>>(
        "std::is_permutation(deque<int>) (3leg) (" + comment + ")", std_is_permutation_3leg);
    bm.template operator()<std::list<int>>(
        "std::is_permutation(list<int>) (3leg) (" + comment + ")", std_is_permutation_3leg);

    // std::is_permutation(it, it, it, pred)
    bm.template operator()<std::vector<int>>(
        "std::is_permutation(vector<int>) (3leg, pred) (" + comment + ")", std_is_permutation_3leg_pred);
    bm.template operator()<std::deque<int>>(
        "std::is_permutation(deque<int>) (3leg, pred) (" + comment + ")", std_is_permutation_3leg_pred);
    bm.template operator()<std::list<int>>(
        "std::is_permutation(list<int>) (3leg, pred) (" + comment + ")", std_is_permutation_3leg_pred);

    // {std,ranges}::is_permutation(it, it, it, it)
    bm.template operator()<std::vector<int>>(
        "std::is_permutation(vector<int>) (4leg) (" + comment + ")", std_is_permutation_4leg);
    bm.template operator()<std::deque<int>>(
        "std::is_permutation(deque<int>) (4leg) (" + comment + ")", std_is_permutation_4leg);
    bm.template operator()<std::list<int>>(
        "std::is_permutation(list<int>) (4leg) (" + comment + ")", std_is_permutation_4leg);

    // {std,ranges}::is_permutation(it, it, it, it, pred)
    bm.template operator()<std::vector<int>>(
        "std::is_permutation(vector<int>) (4leg, pred) (" + comment + ")", std_is_permutation_4leg_pred);
    bm.template operator()<std::deque<int>>(
        "std::is_permutation(deque<int>) (4leg, pred) (" + comment + ")", std_is_permutation_4leg_pred);
    bm.template operator()<std::list<int>>(
        "std::is_permutation(list<int>) (4leg, pred) (" + comment + ")", std_is_permutation_4leg_pred);
  };

  // Benchmark {std,ranges}::is_permutation where both sequences share a common prefix (this can be optimized).
  {
    auto bm = []<class Container>(std::string name, auto is_permutation) {
      benchmark::RegisterBenchmark(
          name,
          [is_permutation](auto& st) {
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
              auto result = is_permutation(c1.begin(), c1.end(), c2.begin(), c2.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1024)
          ->Arg(8192);
    };
    register_benchmarks(bm, "common prefix");
  }

  // Benchmark {std,ranges}::is_permutation on fully shuffled sequences.
  {
    auto bm = []<class Container>(std::string name, auto is_permutation) {
      benchmark::RegisterBenchmark(
          name,
          [is_permutation](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            std::vector<ValueType> data;
            std::generate_n(std::back_inserter(data), size, [] { return Generate<ValueType>::random(); });
            Container c1(data.begin(), data.end());

            std::mt19937 rng;
            std::shuffle(data.begin(), data.end(), rng);
            Container c2(data.begin(), data.end());

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c1);
              benchmark::DoNotOptimize(c2);
              auto result = is_permutation(c1.begin(), c1.end(), c2.begin(), c2.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1024); // this one is very slow, no need for large sequences
    };
    register_benchmarks(bm, "shuffled");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
