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

int main(int argc, char** argv) {
  auto std_shuffle = [](auto first, auto last, auto& rng) { return std::shuffle(first, last, rng); };

  // {std,ranges}::shuffle(normal container)
  {
    auto bm = []<class Container>(std::string name, auto shuffle) {
      benchmark::RegisterBenchmark(
          name,
          [shuffle](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c;
            std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });
            std::mt19937 rng;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              shuffle(c.begin(), c.end(), rng);
              benchmark::DoNotOptimize(c);
            }
          })
          ->Arg(32)
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::shuffle(vector<int>)", std_shuffle);
    bm.operator()<std::deque<int>>("std::shuffle(deque<int>)", std_shuffle);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
