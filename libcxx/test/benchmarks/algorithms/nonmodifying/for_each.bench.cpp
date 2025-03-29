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
          ->Arg(1024)
          ->Arg(4096)
          ->Arg(8192)
          ->Arg(1 << 14)
          ->Arg(1 << 16)
          ->Arg(1 << 18);
    };
    bm.operator()<std::vector<char>>("std::for_each(vector<char>)", std_for_each);
    bm.operator()<std::deque<char>>("std::for_each(deque<char>)", std_for_each);
    bm.operator()<std::list<char>>("std::for_each(list<char>)", std_for_each);
    bm.operator()<std::vector<char>>("rng::for_each(vector<char>)", std::ranges::for_each);
    bm.operator()<std::deque<char>>("rng::for_each(deque<char>)", std::ranges::for_each);
    bm.operator()<std::list<char>>("rng::for_each(list<char>)", std::ranges::for_each);

    bm.operator()<std::vector<short>>("std::for_each(vector<short>)", std_for_each);
    bm.operator()<std::deque<short>>("std::for_each(deque<short>)", std_for_each);
    bm.operator()<std::list<short>>("std::for_each(list<short>)", std_for_each);
    bm.operator()<std::vector<short>>("rng::for_each(vector<short>)", std::ranges::for_each);
    bm.operator()<std::deque<short>>("rng::for_each(deque<short>)", std::ranges::for_each);
    bm.operator()<std::list<short>>("rng::for_each(list<short>)", std::ranges::for_each);

    bm.operator()<std::vector<int>>("std::for_each(vector<int>)", std_for_each);
    bm.operator()<std::deque<int>>("std::for_each(deque<int>)", std_for_each);
    bm.operator()<std::list<int>>("std::for_each(list<int>)", std_for_each);
    bm.operator()<std::vector<int>>("rng::for_each(vector<int>)", std::ranges::for_each);
    bm.operator()<std::deque<int>>("rng::for_each(deque<int>)", std::ranges::for_each);
    bm.operator()<std::list<int>>("rng::for_each(list<int>)", std::ranges::for_each);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
