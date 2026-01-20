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
#include "count_new.h"

int main(int argc, char** argv) {
  auto std_stable_sort = [](auto first, auto last) { return std::stable_sort(first, last); };

  // Benchmark {std,ranges}::stable_sort on various types of data
  //
  // We perform this benchmark in a batch because we need to restore the
  // state of the container after the operation.
  //
  // Also note that we intentionally don't benchmark the predicated version of the algorithm
  // because that makes the benchmark run too slowly.
  {
    auto bm = []<class Container>(std::string name, auto stable_sort, auto generate_data) {
      benchmark::RegisterBenchmark(
          name,
          [stable_sort, generate_data](auto& st) {
            std::size_t const size          = st.range(0);
            constexpr std::size_t BatchSize = 32;
            using ValueType                 = typename Container::value_type;
            std::vector<ValueType> data     = generate_data(size);
            std::array<Container, BatchSize> c;
            std::fill_n(c.begin(), BatchSize, Container(data.begin(), data.end()));

            while (st.KeepRunningBatch(BatchSize)) {
              for (std::size_t i = 0; i != BatchSize; ++i) {
                benchmark::DoNotOptimize(c[i]);
                stable_sort(c[i].begin(), c[i].end());
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
      bm.operator()<std::vector<int>>(name("std::stable_sort(vector<int>)"), std_stable_sort, generate);
      bm.operator()<std::vector<support::NonIntegral>>(
          name("std::stable_sort(vector<NonIntegral>)"), std_stable_sort, gen2);
      bm.operator()<std::deque<int>>(name("std::stable_sort(deque<int>)"), std_stable_sort, generate);
    };

    register_bm(support::quicksort_adversarial_data<int>, "qsort adversarial");
    register_bm(support::ascending_sorted_data<int>, "ascending");
    register_bm(support::descending_sorted_data<int>, "descending");
    register_bm(support::pipe_organ_data<int>, "pipe-organ");
    register_bm(support::heap_data<int>, "heap");
    register_bm(support::shuffled_data<int>, "shuffled");
    register_bm(support::single_element_data<int>, "repeated");
  }

  // Benchmark {std,ranges}::stable_sort when memory allocation fails. The algorithm must fall back to
  // a different algorithm that has different complexity guarantees.
  {
    auto bm = []<class Container>(std::string name, auto stable_sort, auto generate_data) {
      benchmark::RegisterBenchmark(
          name,
          [stable_sort, generate_data](auto& st) {
            std::size_t const size          = st.range(0);
            constexpr std::size_t BatchSize = 32;
            using ValueType                 = typename Container::value_type;
            std::vector<ValueType> data     = generate_data(size);
            std::array<Container, BatchSize> c;
            std::fill_n(c.begin(), BatchSize, Container(data.begin(), data.end()));

            while (st.KeepRunningBatch(BatchSize)) {
              for (std::size_t i = 0; i != BatchSize; ++i) {
                benchmark::DoNotOptimize(c[i]);
                // Disable the ability to allocate memory inside this block
                globalMemCounter.throw_after = 0;

                stable_sort(c[i].begin(), c[i].end());
                benchmark::DoNotOptimize(c[i]);

                globalMemCounter.reset();
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
      auto name = [variant](std::string op) { return op + " (alloc fails, " + variant + ")"; };
      bm.operator()<std::vector<int>>(name("std::stable_sort(vector<int>)"), std_stable_sort, generate);
      bm.operator()<std::vector<support::NonIntegral>>(
          name("std::stable_sort(vector<NonIntegral>)"), std_stable_sort, gen2);
      bm.operator()<std::deque<int>>(name("std::stable_sort(deque<int>)"), std_stable_sort, generate);
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
