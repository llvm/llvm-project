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
  auto std_partition_point = [](auto first, auto last, auto pred) { return std::partition_point(first, last, pred); };

  auto bm = []<class Container>(std::string name, auto partition_point) {
    benchmark::RegisterBenchmark(
        name,
        [partition_point](auto& st) {
          std::size_t const size = st.range(0);
          using ValueType        = typename Container::value_type;
          Container c;
          std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

          // Partition the container in two equally-sized halves. Based on experimentation, the running
          // time of the algorithm doesn't change much depending on the size of the halves.
          ValueType median = compute_median(c.begin(), c.end());
          auto pred        = [median](auto const& element) { return element < median; };
          std::partition(c.begin(), c.end(), pred);
          assert(std::is_partitioned(c.begin(), c.end(), pred));

          for ([[maybe_unused]] auto _ : st) {
            benchmark::DoNotOptimize(c);
            auto result = partition_point(c.begin(), c.end(), pred);
            benchmark::DoNotOptimize(result);
          }
        })
        ->Arg(32)
        ->Arg(50) // non power-of-two
        ->Arg(1024)
        ->Arg(8192);
  };

  // std::partition_point
  bm.operator()<std::vector<int>>("std::partition_point(vector<int>)", std_partition_point);
  bm.operator()<std::deque<int>>("std::partition_point(deque<int>)", std_partition_point);
  bm.operator()<std::list<int>>("std::partition_point(list<int>)", std_partition_point);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
