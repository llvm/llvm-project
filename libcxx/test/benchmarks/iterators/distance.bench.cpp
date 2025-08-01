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
#include <ranges>
#include <vector>

#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  auto std_distance = [](auto first, auto last) { return std::distance(first, last); };

  // {std,ranges}::distance(std::deque)
  {
    auto bm = [](std::string name, auto distance) {
      benchmark::RegisterBenchmark(
          name,
          [distance](auto& st) {
            std::size_t const size = st.range(0);
            std::deque<int> c(size, 1);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = distance(c.begin(), c.end());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(4096)
          ->Arg(8192);
    };
    bm.operator()("std::distance(deque<int>)", std_distance);
    bm.operator()("rng::distance(deque<int>)", std::ranges::distance);
  }

  // {std,ranges}::distance(std::join_view)
  {
    auto bm = []<class Container>(std::string name, auto distance, std::size_t seg_size) {
      benchmark::RegisterBenchmark(
          name,
          [distance, seg_size](auto& st) {
            std::size_t const size     = st.range(0);
            std::size_t const segments = (size + seg_size - 1) / seg_size;
            Container c(segments);
            for (std::size_t i = 0, n = size; i < segments; ++i, n -= seg_size) {
              c[i].resize(std::min(seg_size, n));
            }

            auto view  = c | std::views::join;
            auto first = view.begin();
            auto last  = view.end();

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = distance(first, last);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(4096)
          ->Arg(8192);
    };
    bm.operator()<std::vector<std::vector<int>>>("std::distance(join_view(vector<vector<int>>))", std_distance, 256);
    bm.operator()<std::vector<std::vector<int>>>(
        "rng::distance(join_view(vector<vector<int>>)", std::ranges::distance, 256);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
