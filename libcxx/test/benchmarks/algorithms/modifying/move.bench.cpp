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
  auto std_move = [](auto first, auto last, auto out) { return std::move(first, last, out); };

  // {std,ranges}::move(normal container)
  {
    auto bm = []<class Container>(std::string name, auto move) {
      benchmark::RegisterBenchmark(name, [move](auto& st) {
        std::size_t const n = st.range(0);
        using ValueType     = typename Container::value_type;
        Container c1(n), c2(n);
        std::generate_n(c1.begin(), n, [] { return Generate<ValueType>::random(); });

        Container* in  = &c1;
        Container* out = &c2;
        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c1);
          benchmark::DoNotOptimize(c2);
          auto result = move(in->begin(), in->end(), out->begin());
          benchmark::DoNotOptimize(result);
          std::swap(in, out);
        }
      })->Range(8, 1 << 20);
    };
    bm.operator()<std::vector<int>>("std::move(vector<int>)", std_move);
    bm.operator()<std::deque<int>>("std::move(deque<int>)", std_move);
    bm.operator()<std::list<int>>("std::move(list<int>)", std_move);
    bm.operator()<std::vector<int>>("rng::move(vector<int>)", std::ranges::move);
    bm.operator()<std::deque<int>>("rng::move(deque<int>)", std::ranges::move);
    bm.operator()<std::list<int>>("rng::move(list<int>)", std::ranges::move);
  }

  // {std,ranges}::move(vector<bool>)
  {
    auto bm = []<bool Aligned>(std::string name, auto move) {
      benchmark::RegisterBenchmark(name, [move](auto& st) {
        std::size_t const n = st.range(0);
        std::vector<bool> c1(n, true);
        std::vector<bool> c2(n, false);

        std::vector<bool>* in  = &c1;
        std::vector<bool>* out = &c2;
        for (auto _ : st) {
          auto first1 = in->begin();
          auto last1  = in->end();
          auto first2 = out->begin();
          if constexpr (Aligned) {
            benchmark::DoNotOptimize(move(first1, last1, first2));
          } else {
            benchmark::DoNotOptimize(move(first1 + 4, last1, first2));
          }
          std::swap(in, out);
          benchmark::DoNotOptimize(in);
          benchmark::DoNotOptimize(out);
        }
      })->Range(64, 1 << 20);
    };
    bm.operator()<true>("std::move(vector<bool>) (aligned)", std_move);
    bm.operator()<false>("std::move(vector<bool>) (unaligned)", std_move);
#if TEST_STD_VER >= 23 // vector<bool>::iterator is not an output_iterator before C++23
    bm.operator()<true>("rng::move(vector<bool>) (aligned)", std::ranges::move);
    bm.operator()<false>("rng::move(vector<bool>) (unaligned)", std::ranges::move);
#endif
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
