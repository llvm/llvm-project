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
  auto std_for_each = [](auto first, auto last, auto f) { return std::for_each(first, last, f); };

  // {std,ranges}::for_each
  {
    auto bm = []<class Container>(std::string name, auto for_each) {
      using ElemType = typename Container::value_type;
      benchmark::RegisterBenchmark(
          name,
          [for_each](auto& st) {
            std::size_t const size = st.range(0);
            Container c(size, 1);
            auto first = c.begin();
            auto last  = c.end();

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = for_each(first, last, [](ElemType& x) { x = std::clamp<ElemType>(x, 10, 100); });
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::for_each(vector<int>)", std_for_each);
    bm.operator()<std::deque<int>>("std::for_each(deque<int>)", std_for_each);
    bm.operator()<std::list<int>>("std::for_each(list<int>)", std_for_each);
    bm.operator()<std::vector<int>>("rng::for_each(vector<int>)", std::ranges::for_each);
    bm.operator()<std::deque<int>>("rng::for_each(deque<int>)", std::ranges::for_each);
    bm.operator()<std::list<int>>("rng::for_each(list<int>)", std::ranges::for_each);
  }

  // {std,ranges}::for_each for join_view
  {
    auto bm = []<class Container>(std::string name, auto for_each) {
      using C1       = typename Container::value_type;
      using ElemType = typename C1::value_type;

      benchmark::RegisterBenchmark(
          name,
          [for_each](auto& st) {
            std::size_t const size     = st.range(0);
            std::size_t const seg_size = 256;
            std::size_t const segments = (size + seg_size - 1) / seg_size;
            Container c(segments);
            for (std::size_t i = 0, n = size; i < segments; ++i, n -= seg_size) {
              c[i].resize(std::min(seg_size, n), ElemType(1));
            }

            auto view  = c | std::views::join;
            auto first = view.begin();
            auto last  = view.end();

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = for_each(first, last, [](ElemType& x) { x = std::clamp<ElemType>(x, 10, 100); });
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192);
    };
    bm.operator()<std::vector<std::vector<int>>>("std::for_each(join_view(vector<vector<int>>))", std_for_each);
    bm.operator()<std::vector<std::vector<int>>>("rng::for_each(join_view(vector<vector<int>>)", std::ranges::for_each);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
