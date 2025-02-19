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

// Benchmark copying one out of two element, in alternance. This is basically
// the worst case for this algorithm, I don't think there are many optimizations
// that can be applied in this case.
template <class Container, class Operation>
void bm_copy_every_other_element(std::string operation_name, Operation copy_if) {
  auto bench = [copy_if](auto& st) {
    std::size_t const n = st.range(0);
    using ValueType     = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), n, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> out(n);

    for ([[maybe_unused]] auto _ : st) {
      bool do_copy = false;
      auto pred    = [&do_copy](auto& element) {
        benchmark::DoNotOptimize(element);
        do_copy = !do_copy;
        return do_copy;
      };
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(out);
      auto result = copy_if(c.begin(), c.end(), out.begin(), pred);
      benchmark::DoNotOptimize(result);
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Range(8, 1 << 20);
}

// Copy the full range.
template <class Container, class Operation>
void bm_copy_entire_range(std::string operation_name, Operation copy_if) {
  auto bench = [copy_if](auto& st) {
    std::size_t const n = st.range(0);
    using ValueType     = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), n, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> out(n);

    for ([[maybe_unused]] auto _ : st) {
      auto pred = [](auto& element) {
        benchmark::DoNotOptimize(element);
        return true;
      };
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(out);
      auto result = copy_if(c.begin(), c.end(), out.begin(), pred);
      benchmark::DoNotOptimize(result);
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Range(8, 1 << 20);
}

int main(int argc, char** argv) {
  auto std_copy_if    = [](auto first, auto last, auto out, auto pred) { return std::copy_if(first, last, out, pred); };
  auto ranges_copy_if = std::ranges::copy_if;

  // std::copy_if
  bm_copy_every_other_element<std::vector<int>>("std::copy_if(vector<int>) (every other)", std_copy_if);
  bm_copy_every_other_element<std::deque<int>>("std::copy_if(deque<int>) (every other)", std_copy_if);
  bm_copy_every_other_element<std::list<int>>("std::copy_if(list<int>) (every other)", std_copy_if);

  bm_copy_entire_range<std::vector<int>>("std::copy_if(vector<int>) (entire range)", std_copy_if);
  bm_copy_entire_range<std::deque<int>>("std::copy_if(deque<int>) (entire range)", std_copy_if);
  bm_copy_entire_range<std::list<int>>("std::copy_if(list<int>) (entire range)", std_copy_if);

  // ranges::copy
  bm_copy_every_other_element<std::vector<int>>("ranges::copy_if(vector<int>) (every other)", ranges_copy_if);
  bm_copy_every_other_element<std::deque<int>>("ranges::copy_if(deque<int>) (every other)", ranges_copy_if);
  bm_copy_every_other_element<std::list<int>>("ranges::copy_if(list<int>) (every other)", ranges_copy_if);

  bm_copy_entire_range<std::vector<int>>("ranges::copy_if(vector<int>) (entire range)", ranges_copy_if);
  bm_copy_entire_range<std::deque<int>>("ranges::copy_if(deque<int>) (entire range)", ranges_copy_if);
  bm_copy_entire_range<std::list<int>>("ranges::copy_if(list<int>) (entire range)", ranges_copy_if);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
