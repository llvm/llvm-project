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
  auto std_copy_backward = [](auto first, auto last, auto out) { return std::copy_backward(first, last, out); };

  // {std,ranges}::copy_n(normal container)
  {
    auto bm = []<class InputContainer, class OutputContainer>(std::string name, auto copy_backward) {
      benchmark::RegisterBenchmark(name, [copy_backward](auto& st) TEST_ALIGN_BENCHMARK {
        std::size_t const n = st.range(0);
        using ValueType     = typename InputContainer::value_type;
        InputContainer c;
        std::generate_n(std::back_inserter(c), n, [] { return Generate<ValueType>::random(); });

        OutputContainer out(n);

        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(out);
          auto result = copy_backward(c.begin(), c.end(), out.end());
          benchmark::DoNotOptimize(result);
        }
      })->Range(8, 1 << 20);
    };
    // clang-format off
    bm.operator()<std::vector<int>, std::vector<int>>("std::copy_backward(vector<int>, vector<int>::iterator)", std_copy_backward);
    bm.operator()<std::vector<int>, std::deque<int>>("std::copy_backward(vector<int>, deque<int>::iterator)", std_copy_backward);
    bm.operator()<std::deque<int>, std::vector<int>>("std::copy_backward(deque<int>, vector<int>::iterator)", std_copy_backward);
    bm.operator()<std::deque<int>, std::deque<int>>("std::copy_backward(deque<int>, deque<int>::iterator)", std_copy_backward);
    bm.operator()<std::list<int>, std::vector<int>>("std::copy_backward(list<int>, vector<int>::iterator)", std_copy_backward);
    // clang-format on
  }

  // {std,ranges}::copy_n(vector<bool>)
  {
    auto bm = []<bool Aligned>(std::string name, auto copy_backward) {
      benchmark::RegisterBenchmark(name, [copy_backward](auto& st) TEST_ALIGN_BENCHMARK {
        std::size_t const n = st.range(0);
        std::vector<bool> in(n, true);
        std::vector<bool> out(Aligned ? n : n + 8);
        benchmark::DoNotOptimize(&in);
        auto first = in.begin();
        auto last  = in.end();
        auto dst   = Aligned ? out.end() : out.end() - 4;
        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(in);
          benchmark::DoNotOptimize(out);
          auto result = copy_backward(first, last, dst);
          benchmark::DoNotOptimize(result);
        }
      })->Range(64, 1 << 20);
    };
    bm.operator()<true>("std::copy_backward(vector<bool>, vector<bool>::iterator) (aligned)", std_copy_backward);
    bm.operator()<false>("std::copy_backward(vector<bool>, vector<bool>::iterator) (unaligned)", std_copy_backward);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
