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
#include <type_traits>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"
#include "test_macros.h"

int main(int argc, char** argv) {
  auto std_move = [](auto first, auto last, auto out) { return std::move(first, last, out); };

  // {std,ranges}::move(normal container)
  {
    auto bm = []<class InputContainer, class OutputContainer>(std::string name, auto move) {
      benchmark::RegisterBenchmark(name, [move](auto& st) TEST_ALIGN_BENCHMARK {
        std::size_t const size = st.range(0);
        using ValueType        = typename InputContainer::value_type;
        InputContainer in;
        std::generate_n(std::back_inserter(in), size, [] { return Generate<ValueType>::random(); });

        OutputContainer out(size);

        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(in);
          benchmark::DoNotOptimize(out);
          static_assert(std::is_trivially_move_assignable_v<ValueType>, "avoid double moves");
          auto result = move(in.begin(), in.end(), out.begin());
          benchmark::DoNotOptimize(result);
        }
      })->Range(8, 1 << 20);
    };
    bm.operator()<std::vector<int>, std::vector<int>>("std::move(vector<int>, vector<int>::iterator)", std_move);
    bm.operator()<std::vector<int>, std::deque<int>>("std::move(vector<int>, deque<int>::iterator)", std_move);
    bm.operator()<std::deque<int>, std::vector<int>>("std::move(deque<int>, vector<int>::iterator)", std_move);
    bm.operator()<std::deque<int>, std::deque<int>>("std::move(deque<int>, deque<int>::iterator)", std_move);
    bm.operator()<std::list<int>, std::vector<int>>("std::move(list<int>, vector<int>::iterator)", std_move);
  }

  // {std,ranges}::move(vector<bool>)
  {
    auto bm = []<bool Aligned>(std::string name, auto move) {
      benchmark::RegisterBenchmark(name, [move](auto& st) TEST_ALIGN_BENCHMARK {
        std::size_t const size = st.range(0);
        std::vector<bool> c1(size, true);
        std::vector<bool> c2(size, false);

        std::vector<bool>* in  = &c1;
        std::vector<bool>* out = &c2;
        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(in);
          benchmark::DoNotOptimize(out);
          auto first  = Aligned ? in->begin() : in->begin() + 4;
          auto result = move(first, in->end(), out->begin());
          benchmark::DoNotOptimize(result);
          std::swap(in, out);
        }
      })->Range(64, 1 << 20);
    };
    bm.operator()<true>("std::move(vector<bool>, vector<bool>::iterator) (aligned)", std_move);
    bm.operator()<false>("std::move(vector<bool>, vector<bool>::iterator) (unaligned)", std_move);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
