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
#include <forward_list>
#include <list>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_lower_bound = [](auto first, auto last, auto const& value) { return std::lower_bound(first, last, value); };
  auto std_lower_bound_pred = [](auto first, auto last, auto const& value) {
    return std::lower_bound(first, last, value, [](auto x, auto y) { return x < y; });
  };
  auto std_upper_bound = [](auto first, auto last, auto const& value) { return std::upper_bound(first, last, value); };
  auto std_upper_bound_pred = [](auto first, auto last, auto const& value) {
    return std::upper_bound(first, last, value, [](auto x, auto y) { return x < y; });
  };

  // Benchmark {lower_bound,upper_bound} on a sorted sequence, looking up a random element that
  // is present in the sequence.
  auto bm = []<class Container>(std::string name, auto lookup) {
    benchmark::RegisterBenchmark(
        name,
        [lookup](auto& st) {
          using ValueType        = typename Container::value_type;
          std::size_t const size = st.range(0);

          // Random sorted data
          std::vector<ValueType> data(size);
          std::generate_n(data.begin(), size, &Generate<ValueType>::random);
          std::sort(data.begin(), data.end());

          // Precompute a bunch of random keys.
          std::vector<ValueType> keys(data);
          std::shuffle(keys.begin(), keys.end(), getRandomEngine());

          Container c(data.begin(), data.end());
          std::size_t pos = 0;
          for (auto _ : st) {
            benchmark::DoNotOptimize(c);
            auto result = lookup(c.begin(), c.end(), keys[pos]);
            benchmark::DoNotOptimize(result);
            if (++pos == keys.size())
              pos = 0;
          }
        })
        ->Arg(8)
        ->Arg(100)
        ->Arg(8192);
  };

  bm.operator()<std::vector<int>>("std::lower_bound(std::vector<int>)", std_lower_bound);
  bm.operator()<std::deque<int>>("std::lower_bound(std::deque<int>)", std_lower_bound);
  bm.operator()<std::list<int>>("std::lower_bound(std::list<int>)", std_lower_bound);
  bm.operator()<std::forward_list<int>>("std::lower_bound(std::forward_list<int>)", std_lower_bound);
  bm.operator()<std::vector<int>>("std::lower_bound(std::vector<int>, pred)", std_lower_bound_pred);
  bm.operator()<std::deque<int>>("std::lower_bound(std::deque<int>, pred)", std_lower_bound_pred);
  bm.operator()<std::list<int>>("std::lower_bound(std::list<int>, pred)", std_lower_bound_pred);
  bm.operator()<std::forward_list<int>>("std::lower_bound(std::forward_list<int>, pred)", std_lower_bound_pred);

  bm.operator()<std::vector<int>>("std::upper_bound(std::vector<int>)", std_upper_bound);
  bm.operator()<std::deque<int>>("std::upper_bound(std::deque<int>)", std_upper_bound);
  bm.operator()<std::list<int>>("std::upper_bound(std::list<int>)", std_upper_bound);
  bm.operator()<std::forward_list<int>>("std::upper_bound(std::forward_list<int>)", std_upper_bound);
  bm.operator()<std::vector<int>>("std::upper_bound(std::vector<int>, pred)", std_upper_bound_pred);
  bm.operator()<std::deque<int>>("std::upper_bound(std::deque<int>, pred)", std_upper_bound_pred);
  bm.operator()<std::list<int>>("std::upper_bound(std::list<int>, pred)", std_upper_bound_pred);
  bm.operator()<std::forward_list<int>>("std::upper_bound(std::forward_list<int>, pred)", std_upper_bound_pred);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
