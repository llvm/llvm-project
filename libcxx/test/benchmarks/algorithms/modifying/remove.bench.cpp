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
  auto std_remove = [](auto first, auto last, auto const& value) { return std::remove(first, last, value); };

  // Benchmark {std,ranges}::remove on a sequence of the form xxxxxxxxxxyyyyyyyyyy
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
            ValueType x = Generate<ValueType>::random();
            ValueType y = random_different_from({x});
            for (std::size_t i = 0; i != BatchSize; ++i) {
              c[i]      = Container(size);
              auto half = size / 2;
              std::fill_n(std::fill_n(c[i].begin(), half, x), half, y);
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
                auto half = size / 2;
                std::fill_n(std::fill_n(c[i].begin(), half, x), half, y);
              }
              st.ResumeTiming();
            }
          })
          ->Arg(32)
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::remove(vector<int>) (prefix)", std_remove);
    bm.operator()<std::deque<int>>("std::remove(deque<int>) (prefix)", std_remove);
    bm.operator()<std::list<int>>("std::remove(list<int>) (prefix)", std_remove);
    bm.operator()<std::vector<int>>("rng::remove(vector<int>) (prefix)", std::ranges::remove);
    bm.operator()<std::deque<int>>("rng::remove(deque<int>) (prefix)", std::ranges::remove);
    bm.operator()<std::list<int>>("rng::remove(list<int>) (prefix)", std::ranges::remove);
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
            ValueType x    = Generate<ValueType>::random();
            ValueType y    = random_different_from({x});
            auto alternate = [&](auto out, auto n) {
              for (std::size_t i = 0; i != n; ++i) {
                *out++ = (i % 2 == 0 ? x : y);
              }
            };
            for (std::size_t i = 0; i != BatchSize; ++i) {
              c[i] = Container(size);
              alternate(c[i].begin(), size);
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
                alternate(c[i].begin(), size);
              }
              st.ResumeTiming();
            }
          })
          ->Arg(32)
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::remove(vector<int>) (sprinkled)", std_remove);
    bm.operator()<std::deque<int>>("std::remove(deque<int>) (sprinkled)", std_remove);
    bm.operator()<std::list<int>>("std::remove(list<int>) (sprinkled)", std_remove);
    bm.operator()<std::vector<int>>("rng::remove(vector<int>) (sprinkled)", std::ranges::remove);
    bm.operator()<std::deque<int>>("rng::remove(deque<int>) (sprinkled)", std::ranges::remove);
    bm.operator()<std::list<int>>("rng::remove(list<int>) (sprinkled)", std::ranges::remove);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
