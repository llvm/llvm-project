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
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_is_heap      = [](auto first, auto last) { return std::is_heap(first, last); };
  auto std_is_heap_pred = [](auto first, auto last) {
    return std::is_heap(first, last, [](auto x, auto y) { return x < y; });
  };
  auto std_is_heap_until      = [](auto first, auto last) { return std::is_heap_until(first, last); };
  auto std_is_heap_until_pred = [](auto first, auto last) {
    return std::is_heap_until(first, last, [](auto x, auto y) { return x < y; });
  };

  // Benchmark {std,ranges}::{is_heap,is_heap_until} on a valid heap. This is the worst case since the
  // whole range must be scanned before we can conclude the heap invariant holds.
  {
    auto bm = []<class Container>(std::string name, auto is_heap) {
      benchmark::RegisterBenchmark(
          name,
          [is_heap](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            std::vector<ValueType> data;
            std::generate_n(std::back_inserter(data), size, [] { return Generate<ValueType>::random(); });
            std::make_heap(data.begin(), data.end());

            Container c(data.begin(), data.end());

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = is_heap(c.begin(), c.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::is_heap
    bm.operator()<std::vector<int>>("std::is_heap(vector<int>)", std_is_heap);
    bm.operator()<std::deque<int>>("std::is_heap(deque<int>)", std_is_heap);

    bm.operator()<std::vector<int>>("std::is_heap(vector<int>, pred)", std_is_heap_pred);
    bm.operator()<std::deque<int>>("std::is_heap(deque<int>, pred)", std_is_heap_pred);

    // {std,ranges}::is_heap_until
    bm.operator()<std::vector<int>>("std::is_heap_until(vector<int>)", std_is_heap_until);
    bm.operator()<std::deque<int>>("std::is_heap_until(deque<int>)", std_is_heap_until);

    bm.operator()<std::vector<int>>("std::is_heap_until(vector<int>, pred)", std_is_heap_until_pred);
    bm.operator()<std::deque<int>>("std::is_heap_until(deque<int>, pred)", std_is_heap_until_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
