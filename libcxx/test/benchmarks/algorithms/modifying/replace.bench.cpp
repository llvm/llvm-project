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

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_replace    = [](auto first, auto last, auto old, auto new_) { return std::replace(first, last, old, new_); };
  auto std_replace_if = [](auto first, auto last, auto old, auto new_) {
    auto pred = [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element == old;
    };
    return std::replace_if(first, last, pred, new_);
  };
  auto ranges_replace_if = [](auto first, auto last, auto old, auto new_) {
    auto pred = [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element == old;
    };
    return std::ranges::replace_if(first, last, pred, new_);
  };

  // Create a sequence of the form xxxxxxxxxxyyyyyyyyyy, replace
  // into zzzzzzzzzzzyyyyyyyyyy and then back.
  //
  // This measures the performance of replace() when replacing a large
  // contiguous sequence of equal values.
  {
    auto bm = []<class Container>(std::string name, auto replace) {
      benchmark::RegisterBenchmark(
          name,
          [replace](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            ValueType x = Generate<ValueType>::random();
            ValueType y = random_different_from({x});
            ValueType z = random_different_from({x, y});
            std::fill_n(std::back_inserter(c), size / 2, x);
            std::fill_n(std::back_inserter(c), size / 2, y);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(x);
              benchmark::DoNotOptimize(z);
              replace(c.begin(), c.end(), x, z);
              benchmark::DoNotOptimize(c);
              std::swap(x, z);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::replace
    bm.operator()<std::vector<int>>("std::replace(vector<int>) (prefix)", std_replace);
    bm.operator()<std::deque<int>>("std::replace(deque<int>) (prefix)", std_replace);
    bm.operator()<std::list<int>>("std::replace(list<int>) (prefix)", std_replace);
    bm.operator()<std::vector<int>>("rng::replace(vector<int>) (prefix)", std::ranges::replace);
    bm.operator()<std::deque<int>>("rng::replace(deque<int>) (prefix)", std::ranges::replace);
    bm.operator()<std::list<int>>("rng::replace(list<int>) (prefix)", std::ranges::replace);

    // {std,ranges}::replace_if
    bm.operator()<std::vector<int>>("std::replace_if(vector<int>) (prefix)", std_replace_if);
    bm.operator()<std::deque<int>>("std::replace_if(deque<int>) (prefix)", std_replace_if);
    bm.operator()<std::list<int>>("std::replace_if(list<int>) (prefix)", std_replace_if);
    bm.operator()<std::vector<int>>("rng::replace_if(vector<int>) (prefix)", ranges_replace_if);
    bm.operator()<std::deque<int>>("rng::replace_if(deque<int>) (prefix)", ranges_replace_if);
    bm.operator()<std::list<int>>("rng::replace_if(list<int>) (prefix)", ranges_replace_if);
  }

  // Sprinkle elements to replace inside the range, like xyxyxyxyxyxyxyxyxyxy.
  {
    auto bm = []<class Container>(std::string name, auto replace) {
      benchmark::RegisterBenchmark(
          name,
          [replace](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            ValueType x = Generate<ValueType>::random();
            ValueType y = random_different_from({x});
            ValueType z = random_different_from({x, y});
            for (std::size_t i = 0; i != size; ++i) {
              c.push_back(i % 2 == 0 ? x : y);
            }

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(x);
              benchmark::DoNotOptimize(z);
              replace(c.begin(), c.end(), x, z);
              benchmark::DoNotOptimize(c);
              std::swap(x, z);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::replace
    bm.operator()<std::vector<int>>("std::replace(vector<int>) (sprinkled)", std_replace);
    bm.operator()<std::deque<int>>("std::replace(deque<int>) (sprinkled)", std_replace);
    bm.operator()<std::list<int>>("std::replace(list<int>) (sprinkled)", std_replace);
    bm.operator()<std::vector<int>>("rng::replace(vector<int>) (sprinkled)", std::ranges::replace);
    bm.operator()<std::deque<int>>("rng::replace(deque<int>) (sprinkled)", std::ranges::replace);
    bm.operator()<std::list<int>>("rng::replace(list<int>) (sprinkled)", std::ranges::replace);

    // {std,ranges}::replace_if
    bm.operator()<std::vector<int>>("std::replace_if(vector<int>) (sprinkled)", std_replace_if);
    bm.operator()<std::deque<int>>("std::replace_if(deque<int>) (sprinkled)", std_replace_if);
    bm.operator()<std::list<int>>("std::replace_if(list<int>) (sprinkled)", std_replace_if);
    bm.operator()<std::vector<int>>("rng::replace_if(vector<int>) (sprinkled)", ranges_replace_if);
    bm.operator()<std::deque<int>>("rng::replace_if(deque<int>) (sprinkled)", ranges_replace_if);
    bm.operator()<std::list<int>>("rng::replace_if(list<int>) (sprinkled)", ranges_replace_if);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
