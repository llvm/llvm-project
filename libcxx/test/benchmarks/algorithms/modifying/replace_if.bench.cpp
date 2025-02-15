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
// This measures the performance of replace_if() when replacing a large
// contiguous sequence of equal values.
template <class Container, class Operation>
void bm_prefix(std::string operation_name, Operation replace_if) {
  auto bench = [replace_if](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    ValueType x = Generate<ValueType>::random();
    ValueType y = Generate<ValueType>::random();
    ValueType z = Generate<ValueType>::random();
    std::fill_n(std::back_inserter(c), size / 2, x);
    std::fill_n(std::back_inserter(c), size / 2, y);

    for ([[maybe_unused]] auto _ : st) {
      auto pred = [&x](auto& element) {
        benchmark::DoNotOptimize(element);
        return element == x;
      };
      replace_if(c.begin(), c.end(), pred, z);
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
void bm_sprinkled(std::string operation_name, Operation replace_if) {
  auto bench = [replace_if](auto& st) {
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
      auto pred = [&x](auto& element) {
        benchmark::DoNotOptimize(element);
        return element == x;
      };
      replace_if(c.begin(), c.end(), pred, z);
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
  auto std_replace_if = [](auto first, auto last, auto pred, auto new_) {
    return std::replace_if(first, last, pred, new_);
  };
  auto ranges_replace_if = [](auto first, auto last, auto pred, auto new_) {
    return std::ranges::replace_if(first, last, pred, new_);
  };

  // std::replace_if
  bm_prefix<std::vector<int>>("std::replace_if(vector<int>) (prefix)", std_replace_if);
  bm_sprinkled<std::vector<int>>("std::replace_if(vector<int>) (sprinkled)", std_replace_if);

  bm_prefix<std::deque<int>>("std::replace_if(deque<int>) (prefix)", std_replace_if);
  bm_sprinkled<std::deque<int>>("std::replace_if(deque<int>) (sprinkled)", std_replace_if);

  bm_prefix<std::list<int>>("std::replace_if(list<int>) (prefix)", std_replace_if);
  bm_sprinkled<std::list<int>>("std::replace_if(list<int>) (sprinkled)", std_replace_if);

  // ranges::replace_if
  bm_prefix<std::vector<int>>("ranges::replace_if(vector<int>) (prefix)", ranges_replace_if);
  bm_sprinkled<std::vector<int>>("ranges::replace_if(vector<int>) (sprinkled)", ranges_replace_if);

  bm_prefix<std::deque<int>>("ranges::replace_if(deque<int>) (prefix)", ranges_replace_if);
  bm_sprinkled<std::deque<int>>("ranges::replace_if(deque<int>) (sprinkled)", ranges_replace_if);

  bm_prefix<std::list<int>>("ranges::replace_if(list<int>) (prefix)", ranges_replace_if);
  bm_sprinkled<std::list<int>>("ranges::replace_if(list<int>) (sprinkled)", ranges_replace_if);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
