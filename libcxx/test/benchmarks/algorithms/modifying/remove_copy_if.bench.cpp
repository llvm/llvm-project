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
  auto std_remove_copy_if = [](auto first, auto last, auto out, auto pred) {
    return std::remove_copy_if(first, last, out, pred);
  };

  // Benchmark {std,ranges}::remove_copy_if on a sequence of the form xxxxxxxxxxyyyyyyyyyy
  // where we remove the prefix of x's from the sequence.
  {
    auto bm = []<class Container>(std::string name, auto remove_copy_if) {
      benchmark::RegisterBenchmark(
          name,
          [remove_copy_if](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            ValueType x = Generate<ValueType>::random();
            ValueType y = random_different_from({x});
            std::fill_n(std::back_inserter(c), size / 2, x);
            std::fill_n(std::back_inserter(c), size / 2, y);

            std::vector<ValueType> out(size);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(out);
              auto pred = [&x](auto element) {
                benchmark::DoNotOptimize(element);
                return element == x;
              };
              auto result = remove_copy_if(c.begin(), c.end(), out.begin(), pred);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::remove_copy_if(vector<int>) (prefix)", std_remove_copy_if);
    bm.operator()<std::deque<int>>("std::remove_copy_if(deque<int>) (prefix)", std_remove_copy_if);
    bm.operator()<std::list<int>>("std::remove_copy_if(list<int>) (prefix)", std_remove_copy_if);
    bm.operator()<std::vector<int>>("rng::remove_copy_if(vector<int>) (prefix)", std::ranges::remove_copy_if);
    bm.operator()<std::deque<int>>("rng::remove_copy_if(deque<int>) (prefix)", std::ranges::remove_copy_if);
    bm.operator()<std::list<int>>("rng::remove_copy_if(list<int>) (prefix)", std::ranges::remove_copy_if);
  }

  // Benchmark {std,ranges}::remove_copy_if on a sequence of the form xyxyxyxyxyxyxyxyxyxy
  // where we remove the x's from the sequence.
  {
    auto bm = []<class Container>(std::string name, auto remove_copy_if) {
      benchmark::RegisterBenchmark(
          name,
          [remove_copy_if](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            ValueType x = Generate<ValueType>::random();
            ValueType y = random_different_from({x});
            for (std::size_t i = 0; i != size; ++i) {
              c.push_back(i % 2 == 0 ? x : y);
            }

            std::vector<ValueType> out(size);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(out);
              auto pred = [&](auto element) {
                benchmark::DoNotOptimize(element);
                return element == x;
              };
              auto result = remove_copy_if(c.begin(), c.end(), out.begin(), pred);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::remove_copy_if(vector<int>) (sprinkled)", std_remove_copy_if);
    bm.operator()<std::deque<int>>("std::remove_copy_if(deque<int>) (sprinkled)", std_remove_copy_if);
    bm.operator()<std::list<int>>("std::remove_copy_if(list<int>) (sprinkled)", std_remove_copy_if);
    bm.operator()<std::vector<int>>("rng::remove_copy_if(vector<int>) (sprinkled)", std::ranges::remove_copy_if);
    bm.operator()<std::deque<int>>("rng::remove_copy_if(deque<int>) (sprinkled)", std::ranges::remove_copy_if);
    bm.operator()<std::list<int>>("rng::remove_copy_if(list<int>) (sprinkled)", std::ranges::remove_copy_if);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
