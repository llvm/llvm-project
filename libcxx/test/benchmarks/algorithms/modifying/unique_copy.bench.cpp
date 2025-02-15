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

// Create a sequence of the form xxxxxxxxxxyyyyyyyyyy and unique the
// adjacent equal elements.
template <class Container, class Operation>
void bm_contiguous(std::string operation_name, Operation unique_copy) {
  auto bench = [unique_copy](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c(size);
    ValueType x = Generate<ValueType>::random();
    ValueType y = Generate<ValueType>::random();
    auto half   = size / 2;
    std::fill_n(std::fill_n(c.begin(), half, x), half, y);

    std::vector<ValueType> out(size);

    for ([[maybe_unused]] auto _ : st) {
      auto result = unique_copy(c.begin(), c.end(), out.begin());
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

// Create a sequence of the form xxyyxxyyxxyyxxyyxxyy and unique
// adjacent equal elements.
template <class Container, class Operation>
void bm_sprinkled(std::string operation_name, Operation unique_copy) {
  auto bench = [unique_copy](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c(size);
    ValueType x    = Generate<ValueType>::random();
    ValueType y    = Generate<ValueType>::random();
    auto alternate = [&](auto out, auto n) {
      for (std::size_t i = 0; i != n; i += 2) {
        *out++ = (i % 4 == 0 ? x : y);
        *out++ = (i % 4 == 0 ? x : y);
      }
    };
    alternate(c.begin(), size);

    std::vector<ValueType> out(size);

    for ([[maybe_unused]] auto _ : st) {
      auto result = unique_copy(c.begin(), c.end(), out.begin());
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_unique_copy    = [](auto first, auto last, auto out) { return std::unique_copy(first, last, out); };
  auto ranges_unique_copy = [](auto first, auto last, auto out) { return std::ranges::unique_copy(first, last, out); };

  // std::unique_copy
  bm_contiguous<std::vector<int>>("std::unique_copy(vector<int>) (contiguous)", std_unique_copy);
  bm_sprinkled<std::vector<int>>("std::unique_copy(vector<int>) (sprinkled)", std_unique_copy);

  bm_contiguous<std::deque<int>>("std::unique_copy(deque<int>) (contiguous)", std_unique_copy);
  bm_sprinkled<std::deque<int>>("std::unique_copy(deque<int>) (sprinkled)", std_unique_copy);

  bm_contiguous<std::list<int>>("std::unique_copy(list<int>) (contiguous)", std_unique_copy);
  bm_sprinkled<std::list<int>>("std::unique_copy(list<int>) (sprinkled)", std_unique_copy);

  // ranges::unique_copy
  bm_contiguous<std::vector<int>>("ranges::unique_copy(vector<int>) (contiguous)", ranges_unique_copy);
  bm_sprinkled<std::vector<int>>("ranges::unique_copy(vector<int>) (sprinkled)", ranges_unique_copy);

  bm_contiguous<std::deque<int>>("ranges::unique_copy(deque<int>) (contiguous)", ranges_unique_copy);
  bm_sprinkled<std::deque<int>>("ranges::unique_copy(deque<int>) (sprinkled)", ranges_unique_copy);

  bm_contiguous<std::list<int>>("ranges::unique_copy(list<int>) (contiguous)", ranges_unique_copy);
  bm_sprinkled<std::list<int>>("ranges::unique_copy(list<int>) (sprinkled)", ranges_unique_copy);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
