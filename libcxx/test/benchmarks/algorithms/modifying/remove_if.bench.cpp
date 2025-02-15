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
void bm_prefix(std::string operation_name, Operation remove_if) {
  auto bench = [remove_if](auto& st) {
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

    auto pred = [&](auto& element) {
      benchmark::DoNotOptimize(element);
      return element == x;
    };

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        auto result = remove_if(c[i].begin(), c[i].end(), pred);
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
void bm_sprinkled(std::string operation_name, Operation remove_if) {
  auto bench = [remove_if](auto& st) {
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

    auto pred = [&](auto& element) {
      benchmark::DoNotOptimize(element);
      return element == x;
    };

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        auto result = remove_if(c[i].begin(), c[i].end(), pred);
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
  auto std_remove_if    = [](auto first, auto last, auto pred) { return std::remove_if(first, last, pred); };
  auto ranges_remove_if = [](auto first, auto last, auto pred) { return std::ranges::remove_if(first, last, pred); };

  // std::remove_if
  bm_prefix<std::vector<int>>("std::remove_if(vector<int>) (prefix)", std_remove_if);
  bm_sprinkled<std::vector<int>>("std::remove_if(vector<int>) (sprinkled)", std_remove_if);

  bm_prefix<std::deque<int>>("std::remove_if(deque<int>) (prefix)", std_remove_if);
  bm_sprinkled<std::deque<int>>("std::remove_if(deque<int>) (sprinkled)", std_remove_if);

  bm_prefix<std::list<int>>("std::remove_if(list<int>) (prefix)", std_remove_if);
  bm_sprinkled<std::list<int>>("std::remove_if(list<int>) (sprinkled)", std_remove_if);

  // ranges::remove_if
  bm_prefix<std::vector<int>>("ranges::remove_if(vector<int>) (prefix)", ranges_remove_if);
  bm_sprinkled<std::vector<int>>("ranges::remove_if(vector<int>) (sprinkled)", ranges_remove_if);

  bm_prefix<std::deque<int>>("ranges::remove_if(deque<int>) (prefix)", ranges_remove_if);
  bm_sprinkled<std::deque<int>>("ranges::remove_if(deque<int>) (sprinkled)", ranges_remove_if);

  bm_prefix<std::list<int>>("ranges::remove_if(list<int>) (prefix)", ranges_remove_if);
  bm_sprinkled<std::list<int>>("ranges::remove_if(list<int>) (sprinkled)", ranges_remove_if);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
