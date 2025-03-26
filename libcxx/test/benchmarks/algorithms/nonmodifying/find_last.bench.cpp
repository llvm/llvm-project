//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <cstddef>
#include <deque>
#include <forward_list>
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto ranges_find_last_if = [](auto first, auto last, auto const& value) {
    return std::ranges::find_last_if(first, last, [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element == value;
    });
  };
  auto ranges_find_last_if_not = [](auto first, auto last, auto const& value) {
    return std::ranges::find_last_if_not(first, last, [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element != value;
    });
  };

  // Benchmark ranges::{find_last,find_last_if,find_last_if_not} where the last element
  // is found 10% into the sequence
  {
    auto bm = []<class Container>(std::string name, auto find_last) {
      benchmark::RegisterBenchmark(
          name,
          [find_last](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);

            // put the element we're searching for at 10% of the sequence
            *std::next(c.begin(), size / 10) = y;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(y);
              auto result = find_last(c.begin(), c.end(), y);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };

    // find_last
    bm.operator()<std::vector<char>>("rng::find_last(vector<char>) (bail 10%)", std::ranges::find_last);
    bm.operator()<std::vector<int>>("rng::find_last(vector<int>) (bail 10%)", std::ranges::find_last);
    bm.operator()<std::deque<int>>("rng::find_last(deque<int>) (bail 10%)", std::ranges::find_last);
    bm.operator()<std::list<int>>("rng::find_last(list<int>) (bail 10%)", std::ranges::find_last);
    bm.operator()<std::forward_list<int>>("rng::find_last(forward_list<int>) (bail 10%)", std::ranges::find_last);

    // find_last_if
    bm.operator()<std::vector<char>>("rng::find_last_if(vector<char>) (bail 10%)", ranges_find_last_if);
    bm.operator()<std::vector<int>>("rng::find_last_if(vector<int>) (bail 10%)", ranges_find_last_if);
    bm.operator()<std::deque<int>>("rng::find_last_if(deque<int>) (bail 10%)", ranges_find_last_if);
    bm.operator()<std::list<int>>("rng::find_last_if(list<int>) (bail 10%)", ranges_find_last_if);
    bm.operator()<std::forward_list<int>>("rng::find_last_if(forward_list<int>) (bail 10%)", ranges_find_last_if);

    // find_last_if_not
    bm.operator()<std::vector<char>>("rng::find_last_if_not(vector<char>) (bail 10%)", ranges_find_last_if_not);
    bm.operator()<std::vector<int>>("rng::find_last_if_not(vector<int>) (bail 10%)", ranges_find_last_if_not);
    bm.operator()<std::deque<int>>("rng::find_last_if_not(deque<int>) (bail 10%)", ranges_find_last_if_not);
    bm.operator()<std::list<int>>("rng::find_last_if_not(list<int>) (bail 10%)", ranges_find_last_if_not);
    bm.operator()<std::forward_list<int>>(
        "rng::find_last_if_not(forward_list<int>) (bail 10%)", ranges_find_last_if_not);
  }

  // Benchmark ranges::{find_last,find_last_if,find_last_if_not} where the last element
  // is found 90% into the sequence (i.e. near the end)
  {
    auto bm = []<class Container>(std::string name, auto find_last) {
      benchmark::RegisterBenchmark(
          name,
          [find_last](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);

            // put the element we're searching for at 90% of the sequence
            *std::next(c.begin(), (9 * size) / 10) = y;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(y);
              auto result = find_last(c.begin(), c.end(), y);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    // find_last
    bm.operator()<std::vector<char>>("rng::find_last(vector<char>) (bail 90%)", std::ranges::find_last);
    bm.operator()<std::vector<int>>("rng::find_last(vector<int>) (bail 90%)", std::ranges::find_last);
    bm.operator()<std::deque<int>>("rng::find_last(deque<int>) (bail 90%)", std::ranges::find_last);
    bm.operator()<std::list<int>>("rng::find_last(list<int>) (bail 90%)", std::ranges::find_last);
    bm.operator()<std::forward_list<int>>("rng::find_last(forward_list<int>) (bail 90%)", std::ranges::find_last);

    // find_last_if
    bm.operator()<std::vector<char>>("rng::find_last_if(vector<char>) (bail 90%)", ranges_find_last_if);
    bm.operator()<std::vector<int>>("rng::find_last_if(vector<int>) (bail 90%)", ranges_find_last_if);
    bm.operator()<std::deque<int>>("rng::find_last_if(deque<int>) (bail 90%)", ranges_find_last_if);
    bm.operator()<std::list<int>>("rng::find_last_if(list<int>) (bail 90%)", ranges_find_last_if);
    bm.operator()<std::forward_list<int>>("rng::find_last_if(forward_list<int>) (bail 90%)", ranges_find_last_if);

    // find_last_if_not
    bm.operator()<std::vector<char>>("rng::find_last_if_not(vector<char>) (bail 90%)", ranges_find_last_if_not);
    bm.operator()<std::vector<int>>("rng::find_last_if_not(vector<int>) (bail 90%)", ranges_find_last_if_not);
    bm.operator()<std::deque<int>>("rng::find_last_if_not(deque<int>) (bail 90%)", ranges_find_last_if_not);
    bm.operator()<std::list<int>>("rng::find_last_if_not(list<int>) (bail 90%)", ranges_find_last_if_not);
    bm.operator()<std::forward_list<int>>(
        "rng::find_last_if_not(forward_list<int>) (bail 90%)", ranges_find_last_if_not);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
