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
  auto std_unique = [](auto first, auto last, auto pred) { return std::unique(first, last, pred); };

  // Create a sequence of the form xxxxxxxxxxyyyyyyyyyy and unique the
  // adjacent equal elements.
  //
  // We perform this benchmark in a batch because we need to restore the
  // state of the container after the operation.
  {
    auto bm = []<class Container>(std::string name, auto unique) {
      benchmark::RegisterBenchmark(
          name,
          [unique](auto& st) {
            std::size_t const size          = st.range(0);
            constexpr std::size_t BatchSize = 10;
            using ValueType                 = typename Container::value_type;
            Container c[BatchSize];
            ValueType x = Generate<ValueType>::random();
            ValueType y = Generate<ValueType>::random();
            for (std::size_t i = 0; i != BatchSize; ++i) {
              c[i]      = Container(size);
              auto half = size / 2;
              std::fill_n(std::fill_n(c[i].begin(), half, x), half, y);
            }

            auto pred = [](auto& a, auto& b) {
              benchmark::DoNotOptimize(a);
              benchmark::DoNotOptimize(b);
              return a == b;
            };

            while (st.KeepRunningBatch(BatchSize)) {
              for (std::size_t i = 0; i != BatchSize; ++i) {
                auto result = unique(c[i].begin(), c[i].end(), pred);
                benchmark::DoNotOptimize(result);
                benchmark::DoNotOptimize(c[i]);
                benchmark::ClobberMemory();
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
    bm.operator()<std::vector<int>>("std::unique(vector<int>, pred) (contiguous)", std_unique);
    bm.operator()<std::deque<int>>("std::unique(deque<int>, pred) (contiguous)", std_unique);
    bm.operator()<std::list<int>>("std::unique(list<int>, pred) (contiguous)", std_unique);
    bm.operator()<std::vector<int>>("rng::unique(vector<int>, pred) (contiguous)", std::ranges::unique);
    bm.operator()<std::deque<int>>("rng::unique(deque<int>, pred) (contiguous)", std::ranges::unique);
    bm.operator()<std::list<int>>("rng::unique(list<int>, pred) (contiguous)", std::ranges::unique);
  }

  // Create a sequence of the form xxyyxxyyxxyyxxyyxxyy and unique
  // adjacent equal elements.
  //
  // We perform this benchmark in a batch because we need to restore the
  // state of the container after the operation.
  {
    auto bm = []<class Container>(std::string name, auto unique) {
      benchmark::RegisterBenchmark(
          name,
          [unique](auto& st) {
            std::size_t const size          = st.range(0);
            constexpr std::size_t BatchSize = 10;
            using ValueType                 = typename Container::value_type;
            Container c[BatchSize];
            ValueType x    = Generate<ValueType>::random();
            ValueType y    = Generate<ValueType>::random();
            auto alternate = [&](auto out, auto n) {
              for (std::size_t i = 0; i != n; i += 2) {
                *out++ = (i % 4 == 0 ? x : y);
                *out++ = (i % 4 == 0 ? x : y);
              }
            };
            for (std::size_t i = 0; i != BatchSize; ++i) {
              c[i] = Container(size);
              alternate(c[i].begin(), size);
            }

            auto pred = [](auto& a, auto& b) {
              benchmark::DoNotOptimize(a);
              benchmark::DoNotOptimize(b);
              return a == b;
            };

            while (st.KeepRunningBatch(BatchSize)) {
              for (std::size_t i = 0; i != BatchSize; ++i) {
                auto result = unique(c[i].begin(), c[i].end(), pred);
                benchmark::DoNotOptimize(result);
                benchmark::DoNotOptimize(c[i]);
                benchmark::ClobberMemory();
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
    bm.operator()<std::vector<int>>("std::unique(vector<int>, pred) (sprinkled)", std_unique);
    bm.operator()<std::deque<int>>("std::unique(deque<int>, pred) (sprinkled)", std_unique);
    bm.operator()<std::list<int>>("std::unique(list<int>, pred) (sprinkled)", std_unique);
    bm.operator()<std::vector<int>>("rng::unique(vector<int>, pred) (sprinkled)", std::ranges::unique);
    bm.operator()<std::deque<int>>("rng::unique(deque<int>, pred) (sprinkled)", std::ranges::unique);
    bm.operator()<std::list<int>>("rng::unique(list<int>, pred) (sprinkled)", std::ranges::unique);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
