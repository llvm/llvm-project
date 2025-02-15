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
#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

template <class Container, class Operation>
void bm(std::string operation_name, Operation shuffle) {
  auto bench = [shuffle](auto& st) {
    std::size_t const size = st.range(0);
    using ValueType        = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });
    std::mt19937 rng;

    for ([[maybe_unused]] auto _ : st) {
      shuffle(c.begin(), c.end(), rng);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_shuffle    = [](auto first, auto last, auto& rng) { return std::shuffle(first, last, rng); };
  auto ranges_shuffle = [](auto first, auto last, auto& rng) { return std::ranges::shuffle(first, last, rng); };

  // std::shuffle
  bm<std::vector<int>>("std::shuffle(vector<int>)", std_shuffle);
  bm<std::deque<int>>("std::shuffle(deque<int>)", std_shuffle);

  // ranges::shuffle
  bm<std::vector<int>>("ranges::shuffle(vector<int>)", ranges_shuffle);
  bm<std::deque<int>>("ranges::shuffle(deque<int>)", ranges_shuffle);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
