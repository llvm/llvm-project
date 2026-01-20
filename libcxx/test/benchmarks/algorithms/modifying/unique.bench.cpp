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
  auto std_unique      = [](auto first, auto last) { return std::unique(first, last); };
  auto std_unique_pred = [](auto first, auto last) {
    return std::unique(first, last, [](auto a, auto b) {
      benchmark::DoNotOptimize(a);
      benchmark::DoNotOptimize(b);
      return a == b;
    });
  };

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
                auto result = unique(c[i].begin(), c[i].end());
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
          ->Arg(52) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::unique(it, it)
    bm.operator()<std::vector<int>>("std::unique(vector<int>) (contiguous)", std_unique);
    bm.operator()<std::deque<int>>("std::unique(deque<int>) (contiguous)", std_unique);
    bm.operator()<std::list<int>>("std::unique(list<int>) (contiguous)", std_unique);

    // {std,ranges}::unique(it, it, pred)
    bm.operator()<std::vector<int>>("std::unique(vector<int>, pred) (contiguous)", std_unique_pred);
    bm.operator()<std::deque<int>>("std::unique(deque<int>, pred) (contiguous)", std_unique_pred);
    bm.operator()<std::list<int>>("std::unique(list<int>, pred) (contiguous)", std_unique_pred);
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
            ValueType x   = Generate<ValueType>::random();
            ValueType y   = random_different_from({x});
            auto populate = [&](Container& cont) {
              assert(cont.size() % 4 == 0);
              auto out = cont.begin();
              for (std::size_t i = 0; i != cont.size(); i += 4) {
                *out++ = x;
                *out++ = x;
                *out++ = y;
                *out++ = y;
              }
            };
            for (std::size_t i = 0; i != BatchSize; ++i) {
              c[i] = Container(size);
              populate(c[i]);
            }

            while (st.KeepRunningBatch(BatchSize)) {
              for (std::size_t i = 0; i != BatchSize; ++i) {
                benchmark::DoNotOptimize(c[i]);
                auto result = unique(c[i].begin(), c[i].end());
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
          ->Arg(52) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::unique(it, it)
    bm.operator()<std::vector<int>>("std::unique(vector<int>) (sprinkled)", std_unique);
    bm.operator()<std::deque<int>>("std::unique(deque<int>) (sprinkled)", std_unique);
    bm.operator()<std::list<int>>("std::unique(list<int>) (sprinkled)", std_unique);

    // {std,ranges}::unique(it, it, pred)
    bm.operator()<std::vector<int>>("std::unique(vector<int>, pred) (sprinkled)", std_unique_pred);
    bm.operator()<std::deque<int>>("std::unique(deque<int>, pred) (sprinkled)", std_unique_pred);
    bm.operator()<std::list<int>>("std::unique(list<int>, pred) (sprinkled)", std_unique_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
