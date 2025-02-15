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
template <class Container, class Operation>
void bm_prefix(std::string operation_name, Operation remove_copy_if) {
  auto bench = [remove_copy_if](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    ValueType x = Generate<ValueType>::random();
    ValueType y = Generate<ValueType>::random();
    std::fill_n(std::back_inserter(c), size / 2, x);
    std::fill_n(std::back_inserter(c), size / 2, y);

    auto pred = [&](auto& element) {
      benchmark::DoNotOptimize(element);
      return element == x;
    };

    std::vector<ValueType> out(size);

    for ([[maybe_unused]] auto _ : st) {
      auto result = remove_copy_if(c.begin(), c.end(), out.begin(), pred);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(out);
      benchmark::DoNotOptimize(x);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

// Create a sequence of the form xyxyxyxyxyxyxyxyxyxy and remove
// the x's from it.
template <class Container, class Operation>
void bm_sprinkled(std::string operation_name, Operation remove_copy_if) {
  auto bench = [remove_copy_if](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    ValueType x = Generate<ValueType>::random();
    ValueType y = Generate<ValueType>::random();
    for (std::size_t i = 0; i != size; ++i) {
      c.push_back(i % 2 == 0 ? x : y);
    }

    auto pred = [&](auto& element) {
      benchmark::DoNotOptimize(element);
      return element == x;
    };

    std::vector<ValueType> out(size);

    for ([[maybe_unused]] auto _ : st) {
      auto result = remove_copy_if(c.begin(), c.end(), out.begin(), pred);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(out);
      benchmark::DoNotOptimize(x);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_remove_copy_if = [](auto first, auto last, auto out, auto pred) {
    return std::remove_copy_if(first, last, out, pred);
  };
  auto ranges_remove_copy_if = [](auto first, auto last, auto out, auto pred) {
    return std::ranges::remove_copy_if(first, last, out, pred);
  };

  // std::remove_copy_if
  bm_prefix<std::vector<int>>("std::remove_copy_if(vector<int>) (prefix)", std_remove_copy_if);
  bm_sprinkled<std::vector<int>>("std::remove_copy_if(vector<int>) (sprinkled)", std_remove_copy_if);

  bm_prefix<std::deque<int>>("std::remove_copy_if(deque<int>) (prefix)", std_remove_copy_if);
  bm_sprinkled<std::deque<int>>("std::remove_copy_if(deque<int>) (sprinkled)", std_remove_copy_if);

  bm_prefix<std::list<int>>("std::remove_copy_if(list<int>) (prefix)", std_remove_copy_if);
  bm_sprinkled<std::list<int>>("std::remove_copy_if(list<int>) (sprinkled)", std_remove_copy_if);

  // ranges::remove_copy_if
  bm_prefix<std::vector<int>>("ranges::remove_copy_if(vector<int>) (prefix)", ranges_remove_copy_if);
  bm_sprinkled<std::vector<int>>("ranges::remove_copy_if(vector<int>) (sprinkled)", ranges_remove_copy_if);

  bm_prefix<std::deque<int>>("ranges::remove_copy_if(deque<int>) (prefix)", ranges_remove_copy_if);
  bm_sprinkled<std::deque<int>>("ranges::remove_copy_if(deque<int>) (sprinkled)", ranges_remove_copy_if);

  bm_prefix<std::list<int>>("ranges::remove_copy_if(list<int>) (prefix)", ranges_remove_copy_if);
  bm_sprinkled<std::list<int>>("ranges::remove_copy_if(list<int>) (sprinkled)", ranges_remove_copy_if);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
