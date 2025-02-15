//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <deque>
#include <iterator>
#include <list>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

auto compute_median(auto first, auto last) {
  std::vector v(first, last);
  auto middle = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), middle, v.end());
  return *middle;
}

template <class Container, class Operation>
void bm(std::string operation_name, Operation partition) {
  auto bench = [partition](auto& st) {
    auto const size = st.range(0);
    using ValueType = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> yes(size), no(size);
    ValueType median = compute_median(c.begin(), c.end());
    auto pred1       = [median](auto const& element) { return element < median; };
    auto pred2       = [median](auto const& element) { return element > median; };
    bool toggle      = false;

    for ([[maybe_unused]] auto _ : st) {
      if (toggle) {
        auto result = partition(c.begin(), c.end(), pred1);
        benchmark::DoNotOptimize(result);
      } else {
        auto result = partition(c.begin(), c.end(), pred2);
        benchmark::DoNotOptimize(result);
      }
      toggle = !toggle;

      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_partition    = [](auto first, auto last, auto pred) { return std::partition(first, last, pred); };
  auto ranges_partition = [](auto first, auto last, auto pred) { return std::ranges::partition(first, last, pred); };

  // std::partition
  bm<std::vector<int>>("std::partition(vector<int>)", std_partition);
  bm<std::deque<int>>("std::partition(deque<int>)", std_partition);
  bm<std::list<int>>("std::partition(list<int>)", std_partition);

  // ranges::partition
  bm<std::vector<int>>("ranges::partition(vector<int>)", ranges_partition);
  bm<std::deque<int>>("ranges::partition(deque<int>)", ranges_partition);
  bm<std::list<int>>("ranges::partition(list<int>)", ranges_partition);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
