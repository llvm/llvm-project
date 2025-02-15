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

// Create a sequence of the form xxxxxxxxxxyyyyyyyyyy, replace
// into zzzzzzzzzzzyyyyyyyyyy and then back.
//
// This measures the performance of replace() when replacing a large
// contiguous sequence of equal values.
template <class Container, class Operation>
void bm_prefix(std::string operation_name, Operation replace) {
  auto bench = [replace](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    ValueType x = Generate<ValueType>::random();
    ValueType y = Generate<ValueType>::random();
    ValueType z = Generate<ValueType>::random();
    std::fill_n(std::back_inserter(c), size / 2, x);
    std::fill_n(std::back_inserter(c), size / 2, y);

    for ([[maybe_unused]] auto _ : st) {
      replace(c.begin(), c.end(), x, z);
      std::swap(x, z);
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(z);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

// Sprinkle elements to replace inside the range, like xyxyxyxyxyxyxyxyxyxy.
template <class Container, class Operation>
void bm_sprinkled(std::string operation_name, Operation replace) {
  auto bench = [replace](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    ValueType x = Generate<ValueType>::random();
    ValueType y = Generate<ValueType>::random();
    ValueType z = Generate<ValueType>::random();
    for (std::size_t i = 0; i != size; ++i) {
      c.push_back(i % 2 == 0 ? x : y);
    }

    for ([[maybe_unused]] auto _ : st) {
      replace(c.begin(), c.end(), x, z);
      std::swap(x, z);
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(z);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_replace    = [](auto first, auto last, auto old, auto new_) { return std::replace(first, last, old, new_); };
  auto ranges_replace = [](auto first, auto last, auto old, auto new_) {
    return std::ranges::replace(first, last, old, new_);
  };

  // std::replace
  bm_prefix<std::vector<int>>("std::replace(vector<int>) (prefix)", std_replace);
  bm_sprinkled<std::vector<int>>("std::replace(vector<int>) (sprinkled)", std_replace);

  bm_prefix<std::deque<int>>("std::replace(deque<int>) (prefix)", std_replace);
  bm_sprinkled<std::deque<int>>("std::replace(deque<int>) (sprinkled)", std_replace);

  bm_prefix<std::list<int>>("std::replace(list<int>) (prefix)", std_replace);
  bm_sprinkled<std::list<int>>("std::replace(list<int>) (sprinkled)", std_replace);

  // ranges::replace
  bm_prefix<std::vector<int>>("ranges::replace(vector<int>) (prefix)", ranges_replace);
  bm_sprinkled<std::vector<int>>("ranges::replace(vector<int>) (sprinkled)", ranges_replace);

  bm_prefix<std::deque<int>>("ranges::replace(deque<int>) (prefix)", ranges_replace);
  bm_sprinkled<std::deque<int>>("ranges::replace(deque<int>) (sprinkled)", ranges_replace);

  bm_prefix<std::list<int>>("ranges::replace(list<int>) (prefix)", ranges_replace);
  bm_sprinkled<std::list<int>>("ranges::replace(list<int>) (sprinkled)", ranges_replace);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
