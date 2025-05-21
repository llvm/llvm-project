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
#include <list>
#include <ranges>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  auto std_for_each_n = [](auto first, auto n, auto f) { return std::for_each_n(first, n, f); };

  // std::for_each_n
  {
    auto bm = []<class Container>(std::string name, auto for_each_n) {
      using ElemType = typename Container::value_type;
      benchmark::RegisterBenchmark(
          name,
          [for_each_n](auto& st) {
            std::size_t const n = st.range(0);
            Container c(n, 1);
            auto first = c.begin();

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = for_each_n(first, n, [](ElemType& x) { x = std::clamp<ElemType>(x, 10, 100); });
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(4096)
          ->Arg(8192)
          ->Arg(1 << 14)
          ->Arg(1 << 16)
          ->Arg(1 << 18);
    };
    bm.operator()<std::vector<int>>("std::for_each_n(vector<int>)", std_for_each_n);
    bm.operator()<std::deque<int>>("std::for_each_n(deque<int>)", std_for_each_n);
    bm.operator()<std::list<int>>("std::for_each_n(list<int>)", std_for_each_n);
  }

  // std::for_each_n for join_view
  {
    auto bm = []<class Container>(std::string name, auto for_each_n) {
      using C1       = typename Container::value_type;
      using ElemType = typename C1::value_type;
      benchmark::RegisterBenchmark(
          name,
          [for_each_n](auto& st) {
            std::size_t const size     = st.range(0);
            std::size_t const seg_size = 256;
            std::size_t const segments = (size + seg_size - 1) / seg_size;
            Container c(segments);
            for (std::size_t i = 0, n = size; i < segments; ++i, n -= seg_size) {
              c[i].resize(std::min(seg_size, n), ElemType(1));
            }

            auto view  = c | std::views::join;
            auto first = view.begin();

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = for_each_n(first, size, [](ElemType& x) { x = std::clamp<ElemType>(x, 10, 100); });
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(4096)
          ->Arg(8192)
          ->Arg(1 << 14)
          ->Arg(1 << 16)
          ->Arg(1 << 18);
    };
    bm.operator()<std::vector<std::vector<int>>>("std::for_each_n(join_view(vector<vector<int>>))", std_for_each_n);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
