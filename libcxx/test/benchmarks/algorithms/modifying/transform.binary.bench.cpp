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
    Container c1, c2;
    std::generate_n(std::back_inserter(c1), size, [] { return Generate<ValueType>::random(); });
    std::generate_n(std::back_inserter(c2), size, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> out(size);

    auto f = [](auto& x, auto& y) {
      benchmark::DoNotOptimize(x);
      benchmark::DoNotOptimize(y);
      return x + y;
    };

    for ([[maybe_unused]] auto _ : st) {
      auto result = transform(c1.begin(), c1.end(), c2.begin(), c2.end(), out.begin(), f);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(out);
      benchmark::DoNotOptimize(c1);
      benchmark::DoNotOptimize(c2);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_transform = [](auto first1, auto last1, auto first2, auto, auto out, auto f) {
    return std::transform(first1, last1, first2, out, f);
  };
  auto ranges_transform = [](auto first1, auto last1, auto first2, auto last2, auto out, auto f) {
    return std::ranges::transform(first1, last1, first2, last2, out, f);
  };

  // std::transform
  bm<std::vector<int>>("std::transform(vector<int>, vector<int>)", std_transform);
  bm<std::deque<int>>("std::transform(deque<int>, deque<int>)", std_transform);
  bm<std::list<int>>("std::transform(list<int>, list<int>)", std_transform);

  // ranges::transform
  bm<std::vector<int>>("ranges::transform(vector<int>, vector<int>)", ranges_transform);
  bm<std::deque<int>>("ranges::transform(deque<int>, deque<int>)", ranges_transform);
  bm<std::list<int>>("ranges::transform(list<int>, list<int>)", ranges_transform);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
