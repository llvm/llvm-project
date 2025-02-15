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

// Create a sequence of the form xxxxxxxxxxyyyyyyyyyy and remove
// the prefix of x's from it.
//
// We perform this benchmark in a batch because we need to restore the
// state of the container after the operation.
template <class Container, class Operation>
void bm_prefix(std::string operation_name, Operation remove) {
  auto bench = [remove](auto& st) {
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

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        auto result = remove(c[i].begin(), c[i].end(), x);
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(c[i]);
        benchmark::DoNotOptimize(x);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        auto half = size / 2;
        std::fill_n(std::fill_n(c[i].begin(), half, x), half, y);
      }
      st.ResumeTiming();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

// Create a sequence of the form xyxyxyxyxyxyxyxyxyxy and remove
// the x's from it.
//
// We perform this benchmark in a batch because we need to restore the
// state of the container after the operation.
template <class Container, class Operation>
void bm_sprinkled(std::string operation_name, Operation remove) {
  auto bench = [remove](auto& st) {
    std::size_t const size          = st.range(0);
    constexpr std::size_t BatchSize = 10;
    using ValueType                 = typename Container::value_type;
    Container c[BatchSize];
    ValueType x    = Generate<ValueType>::random();
    ValueType y    = Generate<ValueType>::random();
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
        auto result = remove(c[i].begin(), c[i].end(), x);
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(c[i]);
        benchmark::DoNotOptimize(x);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        alternate(c[i].begin(), size);
      }
      st.ResumeTiming();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_remove    = [](auto first, auto last, auto const& value) { return std::remove(first, last, value); };
  auto ranges_remove = [](auto first, auto last, auto const& value) { return std::ranges::remove(first, last, value); };

  // std::remove
  bm_prefix<std::vector<int>>("std::remove(vector<int>) (prefix)", std_remove);
  bm_sprinkled<std::vector<int>>("std::remove(vector<int>) (sprinkled)", std_remove);

  bm_prefix<std::deque<int>>("std::remove(deque<int>) (prefix)", std_remove);
  bm_sprinkled<std::deque<int>>("std::remove(deque<int>) (sprinkled)", std_remove);

  bm_prefix<std::list<int>>("std::remove(list<int>) (prefix)", std_remove);
  bm_sprinkled<std::list<int>>("std::remove(list<int>) (sprinkled)", std_remove);

  // ranges::remove
  bm_prefix<std::vector<int>>("ranges::remove(vector<int>) (prefix)", ranges_remove);
  bm_sprinkled<std::vector<int>>("ranges::remove(vector<int>) (sprinkled)", ranges_remove);

  bm_prefix<std::deque<int>>("ranges::remove(deque<int>) (prefix)", ranges_remove);
  bm_sprinkled<std::deque<int>>("ranges::remove(deque<int>) (sprinkled)", ranges_remove);

  bm_prefix<std::list<int>>("ranges::remove(list<int>) (prefix)", ranges_remove);
  bm_sprinkled<std::list<int>>("ranges::remove(list<int>) (sprinkled)", ranges_remove);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
