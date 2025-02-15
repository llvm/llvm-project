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

template <class Container, class Operation>
void bm(std::string operation_name, Operation transform) {
  auto bench = [transform](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> out(size);

    auto f = [](auto& element) {
      benchmark::DoNotOptimize(element);
      return element;
    };

    for ([[maybe_unused]] auto _ : st) {
      auto result = transform(c.begin(), c.end(), out.begin(), f);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(out);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_transform    = [](auto first, auto last, auto out, auto f) { return std::transform(first, last, out, f); };
  auto ranges_transform = [](auto first, auto last, auto out, auto f) {
    return std::ranges::transform(first, last, out, f);
  };

  // std::transform
  bm<std::vector<int>>("std::transform(vector<int>) (identity transform)", std_transform);
  bm<std::deque<int>>("std::transform(deque<int>) (identity transform)", std_transform);
  bm<std::list<int>>("std::transform(list<int>) (identity transform)", std_transform);

  // ranges::transform
  bm<std::vector<int>>("ranges::transform(vector<int>) (identity transform)", ranges_transform);
  bm<std::deque<int>>("ranges::transform(deque<int>) (identity transform)", ranges_transform);
  bm<std::list<int>>("ranges::transform(list<int>) (identity transform)", ranges_transform);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
