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
  auto std_remove    = [](auto first, auto last, auto const& value) { return std::remove(first, last, value); };
  auto std_remove_if = [](auto first, auto last, auto const& value) {
    return std::remove_if(first, last, [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element == value;
    });
  };

  // Benchmark {std,ranges}::{remove,remove_if} on a sequence of the form xxxxxxxxxxyyyyyyyyyy
  // where we remove the prefix of x's from the sequence.
  //
  // We perform this benchmark in a batch because we need to restore the
  // state of the container after the operation.
  {
    auto bm = []<class Container>(std::string name, auto remove) {
      benchmark::RegisterBenchmark(
          name,
          [remove](auto& st) {
            std::size_t const size          = st.range(0);
            constexpr std::size_t BatchSize = 10;
            using ValueType                 = typename Container::value_type;
            Container c[BatchSize];
            ValueType x   = Generate<ValueType>::random();
            ValueType y   = random_different_from({x});
            auto populate = [&](Container& cont) {
              auto half = cont.size() / 2;
              std::fill_n(std::fill_n(cont.begin(), half, x), half, y);
            };
            for (std::size_t i = 0; i != BatchSize; ++i) {
              c[i] = Container(size);
              populate(c[i]);
            }

            while (st.KeepRunningBatch(BatchSize)) {
              for (std::size_t i = 0; i != BatchSize; ++i) {
                benchmark::DoNotOptimize(c[i]);
                benchmark::DoNotOptimize(x);
                auto result = remove(c[i].begin(), c[i].end(), x);
                benchmark::DoNotOptimize(result);
              }

              st.PauseTiming();
              for (std::size_t i = 0; i != BatchSize; ++i) {
                populate(c[i]);
              }
              st.ResumeTiming();
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::remove
    bm.operator()<std::vector<int>>("std::remove(vector<int>) (prefix)", std_remove);
    bm.operator()<std::deque<int>>("std::remove(deque<int>) (prefix)", std_remove);
    bm.operator()<std::list<int>>("std::remove(list<int>) (prefix)", std_remove);

    // {std,ranges}::remove_if
    bm.operator()<std::vector<int>>("std::remove_if(vector<int>) (prefix)", std_remove_if);
    bm.operator()<std::deque<int>>("std::remove_if(deque<int>) (prefix)", std_remove_if);
    bm.operator()<std::list<int>>("std::remove_if(list<int>) (prefix)", std_remove_if);
  }

  // Benchmark {std,ranges}::remove on a sequence of the form xyxyxyxyxyxyxyxyxyxy
  // where we remove the x's from the sequence.
  //
  // We perform this benchmark in a batch because we need to restore the
  // state of the container after the operation.
  {
    auto bm = []<class Container>(std::string name, auto remove) {
      benchmark::RegisterBenchmark(
          name,
          [remove](auto& st) {
            std::size_t const size          = st.range(0);
            constexpr std::size_t BatchSize = 10;
            using ValueType                 = typename Container::value_type;
            Container c[BatchSize];
            ValueType x   = Generate<ValueType>::random();
            ValueType y   = random_different_from({x});
            auto populate = [&](Container& cont) {
              auto out = cont.begin();
              for (std::size_t i = 0; i != cont.size(); ++i) {
                *out++ = (i % 2 == 0 ? x : y);
              }
            };
            for (std::size_t i = 0; i != BatchSize; ++i) {
              c[i] = Container(size);
              populate(c[i]);
            }

            while (st.KeepRunningBatch(BatchSize)) {
              for (std::size_t i = 0; i != BatchSize; ++i) {
                benchmark::DoNotOptimize(c[i]);
                benchmark::DoNotOptimize(x);
                auto result = remove(c[i].begin(), c[i].end(), x);
                benchmark::DoNotOptimize(result);
              }

              st.PauseTiming();
              for (std::size_t i = 0; i != BatchSize; ++i) {
                populate(c[i]);
              }
              st.ResumeTiming();
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::remove
    bm.operator()<std::vector<int>>("std::remove(vector<int>) (sprinkled)", std_remove);
    bm.operator()<std::deque<int>>("std::remove(deque<int>) (sprinkled)", std_remove);
    bm.operator()<std::list<int>>("std::remove(list<int>) (sprinkled)", std_remove);

    // {std,ranges}::remove_if
    bm.operator()<std::vector<int>>("std::remove_if(vector<int>) (sprinkled)", std_remove_if);
    bm.operator()<std::deque<int>>("std::remove_if(deque<int>) (sprinkled)", std_remove_if);
    bm.operator()<std::list<int>>("std::remove_if(list<int>) (sprinkled)", std_remove_if);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
