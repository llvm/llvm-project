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
#include <numeric>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_any_of = [](auto first, auto last, auto pred) { return std::any_of(first, last, pred); };
  auto std_all_of = [](auto first, auto last, auto pred) {
    // match semantics of any_of
    return !std::all_of(first, last, [pred](auto x) { return !pred(x); });
  };
  auto std_none_of = [](auto first, auto last, auto pred) {
    // match semantics of any_of
    return !std::none_of(first, last, pred);
  };

  // Benchmark {std,ranges}::{any_of,all_of,none_of} where we process the whole sequence,
  // which is the worst case.
  {
    auto bm = []<class Container>(std::string name, auto any_of) {
      benchmark::RegisterBenchmark(
          name,
          [any_of](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = any_of(c.begin(), c.end(), [&](auto element) {
                benchmark::DoNotOptimize(element);
                return element == y;
              });
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192)
          ->Arg(32768);
    };

    // any_of
    bm.operator()<std::vector<int>>("std::any_of(vector<int>) (process all)", std_any_of);
    bm.operator()<std::deque<int>>("std::any_of(deque<int>) (process all)", std_any_of);
    bm.operator()<std::list<int>>("std::any_of(list<int>) (process all)", std_any_of);

    // all_of
    bm.operator()<std::vector<int>>("std::all_of(vector<int>) (process all)", std_all_of);
    bm.operator()<std::deque<int>>("std::all_of(deque<int>) (process all)", std_all_of);
    bm.operator()<std::list<int>>("std::all_of(list<int>) (process all)", std_all_of);

    // none_of
    bm.operator()<std::vector<int>>("std::none_of(vector<int>) (process all)", std_none_of);
    bm.operator()<std::deque<int>>("std::none_of(deque<int>) (process all)", std_none_of);
    bm.operator()<std::list<int>>("std::none_of(list<int>) (process all)", std_none_of);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
