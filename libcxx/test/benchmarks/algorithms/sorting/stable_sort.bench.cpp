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
#define BENCH(generate_data, name)                                                                                     \
  do {                                                                                                                 \
    auto gen1 = [](auto size) { return generate_data<int>(size); };                                                    \
    auto gen2 = [](auto size) {                                                                                        \
      auto data = generate_data<int>(size);                                                                            \
      std::vector<support::NonIntegral> real_data(data.begin(), data.end());                                           \
      return real_data;                                                                                                \
    };                                                                                                                 \
    bm.operator()<std::vector<int>>("std::stable_sort(vector<int>) (" #name ")", std_stable_sort, gen1);               \
    bm.operator()<std::vector<support::NonIntegral>>(                                                                  \
        "std::stable_sort(vector<NonIntegral>) (" #name ")", std_stable_sort, gen2);                                   \
    bm.operator()<std::deque<int>>("std::stable_sort(deque<int>) (" #name ")", std_stable_sort, gen1);                 \
                                                                                                                       \
    bm.operator()<std::vector<int>>("rng::stable_sort(vector<int>) (" #name ")", std::ranges::stable_sort, gen1);      \
    bm.operator()<std::vector<support::NonIntegral>>(                                                                  \
        "rng::stable_sort(vector<NonIntegral>) (" #name ")", std::ranges::stable_sort, gen2);                          \
    bm.operator()<std::deque<int>>("rng::stable_sort(deque<int>) (" #name ")", std::ranges::stable_sort, gen1);        \
  } while (false)

    BENCH(support::quicksort_adversarial_data, "qsort adversarial");
    BENCH(support::ascending_sorted_data, "ascending");
    BENCH(support::descending_sorted_data, "descending");
    BENCH(support::pipe_organ_data, "pipe-organ");
    BENCH(support::heap_data, "heap");
    BENCH(support::shuffled_data, "shuffled");
    BENCH(support::single_element_data, "repeated");
#undef BENCH
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
#define BENCH(generate_data, name)                                                                                     \
  do {                                                                                                                 \
    auto gen1 = [](auto size) { return generate_data<int>(size); };                                                    \
    auto gen2 = [](auto size) {                                                                                        \
      auto data = generate_data<int>(size);                                                                            \
      std::vector<support::NonIntegral> real_data(data.begin(), data.end());                                           \
      return real_data;                                                                                                \
    };                                                                                                                 \
    bm.operator()<std::vector<int>>("std::stable_sort(vector<int>) (alloc fails, " #name ")", std_stable_sort, gen1);  \
    bm.operator()<std::vector<support::NonIntegral>>(                                                                  \
        "std::stable_sort(vector<NonIntegral>) (alloc fails, " #name ")", std_stable_sort, gen2);                      \
    bm.operator()<std::deque<int>>("std::stable_sort(deque<int>) (alloc fails, " #name ")", std_stable_sort, gen1);    \
                                                                                                                       \
    bm.operator()<std::vector<int>>(                                                                                   \
        "rng::stable_sort(vector<int>) (alloc fails, " #name ")", std::ranges::stable_sort, gen1);                     \
    bm.operator()<std::vector<support::NonIntegral>>(                                                                  \
        "rng::stable_sort(vector<NonIntegral>) (alloc fails, " #name ")", std::ranges::stable_sort, gen2);             \
    bm.operator()<std::deque<int>>(                                                                                    \
        "rng::stable_sort(deque<int>) (alloc fails, " #name ")", std::ranges::stable_sort, gen1);                      \
  } while (false)

    BENCH(support::quicksort_adversarial_data, "qsort adversarial");
    BENCH(support::ascending_sorted_data, "ascending");
    BENCH(support::descending_sorted_data, "descending");
    BENCH(support::pipe_organ_data, "pipe-organ");
    BENCH(support::heap_data, "heap");
    BENCH(support::shuffled_data, "shuffled");
    BENCH(support::single_element_data, "repeated");
#undef BENCH
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
