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

auto compute_median(auto first, auto last) {
  std::vector v(first, last);
  auto middle = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), middle, v.end());
  return *middle;
}

int main(int argc, char** argv) {
  auto std_partition_copy = [](auto first, auto last, auto out_yes, auto out_no, auto pred) {
    return std::partition_copy(first, last, out_yes, out_no, pred);
  };

  auto bm = []<class Container>(std::string name, auto partition_copy) {
    benchmark::RegisterBenchmark(
        name,
        [partition_copy](auto& st) {
          std::size_t const size = st.range(0);
          using ValueType        = typename Container::value_type;
          Container c;
          std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

          std::vector<ValueType> yes(size);
          std::vector<ValueType> no(size);
          ValueType median = compute_median(c.begin(), c.end());
          auto pred        = [median](auto const& element) { return element < median; };

          for ([[maybe_unused]] auto _ : st) {
            benchmark::DoNotOptimize(c);
            benchmark::DoNotOptimize(yes);
            benchmark::DoNotOptimize(no);
            auto result = partition_copy(c.begin(), c.end(), yes.begin(), no.begin(), pred);
            benchmark::DoNotOptimize(result);
          }
        })
        ->Arg(32)
        ->Arg(50) // non power-of-two
        ->Arg(1024)
        ->Arg(8192);
  };

  // std::partition_copy
  bm.operator()<std::vector<int>>("std::partition_copy(vector<int>)", std_partition_copy);
  bm.operator()<std::deque<int>>("std::partition_copy(deque<int>)", std_partition_copy);
  bm.operator()<std::list<int>>("std::partition_copy(list<int>)", std_partition_copy);

  // ranges::partition_copy
  bm.operator()<std::vector<int>>("rng::partition_copy(vector<int>)", std::ranges::partition_copy);
  bm.operator()<std::deque<int>>("rng::partition_copy(deque<int>)", std::ranges::partition_copy);
  bm.operator()<std::list<int>>("rng::partition_copy(list<int>)", std::ranges::partition_copy);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
