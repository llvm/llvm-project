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
  auto std_move_backward = [](auto first, auto last, auto out) { return std::move_backward(first, last, out); };

  // {std,ranges}::move_backward(normal container)
  {
    auto bm = []<class Container>(std::string name, auto move_backward) {
      benchmark::RegisterBenchmark(name, [move_backward](auto& st) {
        std::size_t const n = st.range(0);
        using ValueType     = typename Container::value_type;
        Container c1(n), c2(n);
        std::generate_n(c1.begin(), n, [] { return Generate<ValueType>::random(); });

        Container* in  = &c1;
        Container* out = &c2;
        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c1);
          benchmark::DoNotOptimize(c2);
          auto result = move_backward(in->begin(), in->end(), out->end());
          benchmark::DoNotOptimize(result);
          std::swap(in, out);
        }
      })->Range(8, 1 << 20);
    };
    bm.operator()<std::vector<int>>("std::move_backward(vector<int>)", std_move_backward);
    bm.operator()<std::deque<int>>("std::move_backward(deque<int>)", std_move_backward);
    bm.operator()<std::list<int>>("std::move_backward(list<int>)", std_move_backward);
    bm.operator()<std::vector<int>>("rng::move_backward(vector<int>)", std::ranges::move_backward);
    bm.operator()<std::deque<int>>("rng::move_backward(deque<int>)", std::ranges::move_backward);
    bm.operator()<std::list<int>>("rng::move_backward(list<int>)", std::ranges::move_backward);
  }

  // {std,ranges}::move_backward(vector<bool>)
  {
    auto bm = []<bool Aligned>(std::string name, auto move_backward) {
      benchmark::RegisterBenchmark(name, [move_backward](auto& st) {
        std::size_t const n = st.range(0);
        std::vector<bool> c1(n, true);
        std::vector<bool> c2(n, false);

        std::vector<bool>* in  = &c1;
        std::vector<bool>* out = &c2;
        for (auto _ : st) {
          auto first1 = in->begin();
          auto last1  = in->end();
          auto last2  = out->end();
          if constexpr (Aligned) {
            benchmark::DoNotOptimize(move_backward(first1, last1, last2));
          } else {
            benchmark::DoNotOptimize(move_backward(first1, last1 - 4, last2));
          }
          std::swap(in, out);
          benchmark::DoNotOptimize(in);
          benchmark::DoNotOptimize(out);
        }
      })->Range(64, 1 << 20);
    };
    bm.operator()<true>("std::move_backward(vector<bool>) (aligned)", std_move_backward);
    bm.operator()<false>("std::move_backward(vector<bool>) (unaligned)", std_move_backward);
#if TEST_STD_VER >= 23 // vector<bool>::iterator is not an output_iterator before C++23
    bm.operator()<true>("rng::move_backward(vector<bool>) (aligned)", std::ranges::move_backward);
    bm.operator()<false>("rng::move_backward(vector<bool>) (unaligned)", std::ranges::move_backward);
#endif
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
