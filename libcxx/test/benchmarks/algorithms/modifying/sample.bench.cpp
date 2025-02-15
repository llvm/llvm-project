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
#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

template <class Container, class Operation>
void bm(std::string operation_name, Operation sample) {
  auto bench = [sample](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> out(size);
    auto const n = size / 4; // sample 1/4 of the range
    std::mt19937 rng;

    for ([[maybe_unused]] auto _ : st) {
      auto result = sample(c.begin(), c.end(), out.begin(), n, rng);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(out);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_sample = [](auto first, auto last, auto out, auto n, auto& rng) {
    return std::sample(first, last, out, n, rng);
  };
  auto ranges_sample = [](auto first, auto last, auto out, auto n, auto& rng) {
    return std::ranges::sample(first, last, out, n, rng);
  };

  // std::sample
  bm<std::vector<int>>("std::sample(vector<int>)", std_sample);
  bm<std::deque<int>>("std::sample(deque<int>)", std_sample);
  bm<std::list<int>>("std::sample(list<int>)", std_sample);

  // ranges::sample
  bm<std::vector<int>>("ranges::sample(vector<int>)", ranges_sample);
  bm<std::deque<int>>("ranges::sample(deque<int>)", ranges_sample);
  bm<std::list<int>>("ranges::sample(list<int>)", ranges_sample);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
