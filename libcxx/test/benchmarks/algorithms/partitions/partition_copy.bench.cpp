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
void bm(std::string operation_name, Operation partition_copy) {
  auto bench = [partition_copy](auto& st) {
    auto const size = st.range(0);
    using ValueType = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> yes(size), no(size);
    ValueType median = compute_median(c.begin(), c.end());
    auto pred        = [median](auto const& element) { return element < median; };

    for ([[maybe_unused]] auto _ : st) {
      auto result = partition_copy(c.begin(), c.end(), yes.begin(), no.begin(), pred);
      benchmark::DoNotOptimize(yes);
      benchmark::DoNotOptimize(no);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_partition_copy = [](auto first, auto last, auto out_yes, auto out_no, auto pred) {
    return std::partition_copy(first, last, out_yes, out_no, pred);
  };
  auto ranges_partition_copy = [](auto first, auto last, auto out_yes, auto out_no, auto pred) {
    return std::ranges::partition_copy(first, last, out_yes, out_no, pred);
  };

  // std::partition_copy
  bm<std::vector<int>>("std::partition_copy(vector<int>)", std_partition_copy);
  bm<std::deque<int>>("std::partition_copy(deque<int>)", std_partition_copy);
  bm<std::list<int>>("std::partition_copy(list<int>)", std_partition_copy);

  // ranges::partition_copy
  bm<std::vector<int>>("ranges::partition_copy(vector<int>)", ranges_partition_copy);
  bm<std::deque<int>>("ranges::partition_copy(deque<int>)", ranges_partition_copy);
  bm<std::list<int>>("ranges::partition_copy(list<int>)", ranges_partition_copy);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
