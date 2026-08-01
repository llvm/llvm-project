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
  auto std_replace_copy = [](auto first, auto last, auto out, auto const& old_value, auto const& new_value) {
    return std::replace_copy(first, last, out, old_value, new_value);
  };
  auto std_replace_copy_if = [](auto first, auto last, auto out, auto const& old_value, auto const& new_value) {
    return std::replace_copy_if(first, last, out, [&](auto element) { return element == old_value; }, new_value);
  };

  // Benchmark {std,ranges}::{replace_copy,replace_copy_if} on a sequence of the form xxxxxxxxxxyyyyyyyyyy
  // where we replace the prefix of x's with z's.
  {
    auto bm = []<class Container>(std::string name, auto replace_copy) {
      benchmark::RegisterBenchmark(
          name,
          [replace_copy](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            ValueType x = Generate<ValueType>::random();
            ValueType y = random_different_from({x});
            ValueType z = random_different_from({x, y});
            std::fill_n(std::back_inserter(c), size / 2, x);
            std::fill_n(std::back_inserter(c), size / 2, y);

            std::vector<ValueType> out(size);

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(out);
              benchmark::DoNotOptimize(x);
              benchmark::DoNotOptimize(z);
              auto result = replace_copy(c.begin(), c.end(), out.begin(), x, z);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::replace_copy
    bm.operator()<std::vector<int>>("std::replace_copy(vector<int>) (prefix)", std_replace_copy);
    bm.operator()<std::deque<int>>("std::replace_copy(deque<int>) (prefix)", std_replace_copy);
    bm.operator()<std::list<int>>("std::replace_copy(list<int>) (prefix)", std_replace_copy);

    // {std,ranges}::replace_copy_if
    bm.operator()<std::vector<int>>("std::replace_copy_if(vector<int>) (prefix)", std_replace_copy_if);
    bm.operator()<std::deque<int>>("std::replace_copy_if(deque<int>) (prefix)", std_replace_copy_if);
    bm.operator()<std::list<int>>("std::replace_copy_if(list<int>) (prefix)", std_replace_copy_if);
  }

  // Benchmark {std,ranges}::{replace_copy,replace_copy_if} on a sequence of the form xyxyxyxyxyxyxyxyxyxy
  // where we replace the x's with z's.
  {
    auto bm = []<class Container>(std::string name, auto replace_copy) {
      benchmark::RegisterBenchmark(
          name,
          [replace_copy](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            ValueType x = Generate<ValueType>::random();
            ValueType y = random_different_from({x});
            ValueType z = random_different_from({x, y});
            for (std::size_t i = 0; i != size; ++i) {
              c.push_back(i % 2 == 0 ? x : y);
            }

            std::vector<ValueType> out(size);

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(out);
              benchmark::DoNotOptimize(x);
              benchmark::DoNotOptimize(z);
              auto result = replace_copy(c.begin(), c.end(), out.begin(), x, z);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::replace_copy
    bm.operator()<std::vector<int>>("std::replace_copy(vector<int>) (sprinkled)", std_replace_copy);
    bm.operator()<std::deque<int>>("std::replace_copy(deque<int>) (sprinkled)", std_replace_copy);
    bm.operator()<std::list<int>>("std::replace_copy(list<int>) (sprinkled)", std_replace_copy);

    // {std,ranges}::replace_copy_if
    bm.operator()<std::vector<int>>("std::replace_copy_if(vector<int>) (sprinkled)", std_replace_copy_if);
    bm.operator()<std::deque<int>>("std::replace_copy_if(deque<int>) (sprinkled)", std_replace_copy_if);
    bm.operator()<std::list<int>>("std::replace_copy_if(list<int>) (sprinkled)", std_replace_copy_if);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
