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
#include "test_macros.h"

template <class Container, class Operation>
void bm(std::string operation_name, Operation fill) {
  auto bench = [fill](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    ValueType x            = Generate<ValueType>::random();
    ValueType y            = Generate<ValueType>::random();
    Container c(size, y);

    for ([[maybe_unused]] auto _ : st) {
      fill(c.begin(), c.end(), x);
      std::swap(x, y);
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

template <class Operation>
void bm_vector_bool(std::string operation_name, Operation fill) {
  auto bench = [fill](auto& st) {
    std::size_t const size = st.range(0);
    bool x                 = true;
    bool y                 = false;
    std::vector<bool> c(size, y);

    for ([[maybe_unused]] auto _ : st) {
      fill(c.begin(), c.end(), x);
      std::swap(x, y);
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_fill    = [](auto first, auto last, auto const& value) { return std::fill(first, last, value); };
  auto ranges_fill = [](auto first, auto last, auto const& value) { return std::ranges::fill(first, last, value); };

  // std::fill
  bm<std::vector<int>>("std::fill(vector<int>)", std_fill);
  bm<std::deque<int>>("std::fill(deque<int>)", std_fill);
  bm<std::list<int>>("std::fill(list<int>)", std_fill);
  bm_vector_bool("std::fill(vector<bool>)", std_fill);

  // ranges::fill
  bm<std::vector<int>>("ranges::fill(vector<int>)", ranges_fill);
  bm<std::deque<int>>("ranges::fill(deque<int>)", ranges_fill);
  bm<std::list<int>>("ranges::fill(list<int>)", ranges_fill);
#if TEST_STD_VER >= 23 // vector<bool>::iterator is not an output_iterator before C++23
  bm_vector_bool("ranges::fill(vector<bool>)", ranges_fill);
#endif

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
