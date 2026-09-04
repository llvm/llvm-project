//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "common.h"
#include "test_macros.h"

int main(int argc, char** argv) {
  auto std_nth_element = [](auto first, auto nth, auto last) { return std::nth_element(first, nth, last); };

  // Benchmark std::nth_element on various types of data. We always partition
  // around the middle element of the full range. We don't benchmark
  // std::ranges::nth_element separately because it forwards to the same core.
  //
  // We run this in batches because nth_element mutates the range in place, so we
  // restore the original data after each batch.
  //
  // We intentionally don't benchmark the predicated version of the algorithm
  // because that makes the benchmark run too slowly.
  {
    auto bm = []<class Container>(std::string name, auto nth_element, auto generate_data) {
      benchmark::RegisterBenchmark(
          name,
          [nth_element, generate_data](auto& st) TEST_ALIGN_BENCHMARK {
            std::size_t const size          = st.range(0);
            constexpr std::size_t BatchSize = 32;
            using ValueType                 = typename Container::value_type;
            std::vector<ValueType> data     = generate_data(size);
            std::array<Container, BatchSize> c;
            std::fill_n(c.begin(), BatchSize, Container(data.begin(), data.end()));

            std::size_t const half = size / 2;
            while (st.KeepRunningBatch(BatchSize)) {
              for (std::size_t i = 0; i != BatchSize; ++i) {
                benchmark::DoNotOptimize(c[i]);
                nth_element(c[i].begin(), c[i].begin() + half, c[i].end());
                benchmark::DoNotOptimize(c[i]);
              }

              st.PauseTiming();
              for (std::size_t i = 0; i != BatchSize; ++i) {
                std::copy(data.begin(), data.end(), c[i].begin());
              }
              st.ResumeTiming();
            }
          })
          ->Arg(8)
          ->Arg(1024)
          ->Arg(8192);
    };

    auto register_bm = [&](auto generate, std::string variant) {
      auto gen2 = [generate](auto size) {
        std::vector<int> data = generate(size);
        std::vector<support::NonIntegral> real_data(data.begin(), data.end());
        return real_data;
      };
      auto name = [variant](std::string op) { return op + " (" + variant + ")"; };
      bm.operator()<std::vector<int>>(name("std::nth_element(vector<int>)"), std_nth_element, generate);
      bm.operator()<std::vector<support::NonIntegral>>(
          name("std::nth_element(vector<NonIntegral>)"), std_nth_element, gen2);
      bm.operator()<std::deque<int>>(name("std::nth_element(deque<int>)"), std_nth_element, generate);
    };

    register_bm(support::quicksort_adversarial_data<int>, "qsort adversarial");
    register_bm(support::ascending_sorted_data<int>, "ascending");
    register_bm(support::descending_sorted_data<int>, "descending");
    register_bm(support::pipe_organ_data<int>, "pipe-organ");
    register_bm(support::heap_data<int>, "heap");
    register_bm(support::shuffled_data<int>, "shuffled");
    register_bm(support::single_element_data<int>, "repeated");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
