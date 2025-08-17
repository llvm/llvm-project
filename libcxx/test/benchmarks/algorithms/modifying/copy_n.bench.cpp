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
#include "test_macros.h"

int main(int argc, char** argv) {
  auto std_copy_n = [](auto first, auto n, auto out) { return std::copy_n(first, n, out); };

  // {std,ranges}::copy_n(normal container)
  {
    auto bm = []<class Container>(std::string name, auto copy_n) {
      benchmark::RegisterBenchmark(name, [copy_n](auto& st) {
        std::size_t const n = st.range(0);
        using ValueType     = typename Container::value_type;
        Container c;
        std::generate_n(std::back_inserter(c), n, [] { return Generate<ValueType>::random(); });

        std::vector<ValueType> out(n);

        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(out);
          auto result = copy_n(c.begin(), n, out.begin());
          benchmark::DoNotOptimize(result);
        }
      })->Range(8, 1 << 20);
    };
    bm.operator()<std::vector<int>>("std::copy_n(vector<int>)", std_copy_n);
    bm.operator()<std::deque<int>>("std::copy_n(deque<int>)", std_copy_n);
    bm.operator()<std::list<int>>("std::copy_n(list<int>)", std_copy_n);
    bm.operator()<std::vector<int>>("rng::copy_n(vector<int>)", std::ranges::copy_n);
    bm.operator()<std::deque<int>>("rng::copy_n(deque<int>)", std::ranges::copy_n);
    bm.operator()<std::list<int>>("rng::copy_n(list<int>)", std::ranges::copy_n);
  }

  // {std,ranges}::copy_n(vector<bool>)
  {
    auto bm = []<bool Aligned>(std::string name, auto copy_n) {
      benchmark::RegisterBenchmark(name, [copy_n](auto& st) {
        std::size_t const n = st.range(0);
        std::vector<bool> in(n, true);
        std::vector<bool> out(Aligned ? n : n + 8);
        auto first = in.begin();
        auto dst   = Aligned ? out.begin() : out.begin() + 4;
        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(in);
          benchmark::DoNotOptimize(out);
          auto result = copy_n(first, n, dst);
          benchmark::DoNotOptimize(result);
        }
      })->Range(64, 1 << 20);
    };
    bm.operator()<true>("std::copy_n(vector<bool>) (aligned)", std_copy_n);
    bm.operator()<false>("std::copy_n(vector<bool>) (unaligned)", std_copy_n);
#if TEST_STD_VER >= 23 // vector<bool>::iterator is not an output_iterator before C++23
    bm.operator()<true>("rng::copy_n(vector<bool>) (aligned)", std::ranges::copy_n);
    bm.operator()<false>("rng::copy_n(vector<bool>) (unaligned)", std::ranges::copy_n);
#endif
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
