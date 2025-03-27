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
#include "test_iterators.h"
#include "test_macros.h"

int main(int argc, char** argv) {
  auto std_copy = [](auto first, auto last, auto out) { return std::copy(first, last, out); };

  // {std,ranges}::copy(normal container)
  {
    auto bm = []<class ContainerIn, class ContainerOut = std::vector<typename ContainerIn::value_type>>(
                  std::string name, auto copy) {
      benchmark::RegisterBenchmark(name, [copy](auto& st) {
        std::size_t const n = st.range(0);
        using ValueType     = typename ContainerIn::value_type;
        ContainerIn c;
        std::generate_n(std::back_inserter(c), n, [] { return Generate<ValueType>::random(); });

        ContainerOut out(n);

        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(out);
          auto result = copy(c.begin(), c.end(), out.begin());
          benchmark::DoNotOptimize(result);
        }
      })->Range(8, 1 << 20);
    };
    // Copy from normal containers to vector<int>
    bm.operator()<std::vector<int>>("std::copy(vector<int>)", std_copy);
    bm.operator()<std::deque<int>>("std::copy(deque<int>)", std_copy);
    bm.operator()<std::list<int>>("std::copy(list<int>)", std_copy);
    bm.operator()<std::vector<int>>("rng::copy(vector<int>)", std::ranges::copy);
    bm.operator()<std::deque<int>>("rng::copy(deque<int>)", std::ranges::copy);
    bm.operator()<std::list<int>>("rng::copy(list<int>)", std::ranges::copy);

#if TEST_STD_VER >= 23 // vector<bool>::iterator is not an output_iterator before C++23
    // Copy from normal containers to vector<bool>
    bm.operator()<std::vector<int>, std::vector<bool>>("std::copy(vector<int>, std::vector<bool>)", std_copy);
    bm.operator()<std::deque<int>, std::vector<bool>>("std::copy(deque<int>, std::vector<bool>)", std_copy);
    bm.operator()<std::list<int>, std::vector<bool>>("std::copy(list<int>, std::vector<bool>)", std_copy);
    bm.operator()<std::vector<int>, std::vector<bool>>("rng::copy(vector<int>, std::vector<bool>)", std::ranges::copy);
    bm.operator()<std::deque<int>, std::vector<bool>>("rng::copy(deque<int>, std::vector<bool>)", std::ranges::copy);
    bm.operator()<std::list<int>, std::vector<bool>>("rng::copy(list<int>, std::vector<bool>)", std::ranges::copy);
#endif
  }

  // {std,ranges}::copy(vector<bool>)
  {
    auto bm = []<bool Aligned>(std::string name, auto copy) {
      benchmark::RegisterBenchmark(name, [copy](auto& st) {
        std::size_t const n = st.range(0);
        std::vector<bool> in(n, true);
        std::vector<bool> out(Aligned ? n : n + 8);
        auto first = in.begin();
        auto last  = in.end();
        auto dst   = Aligned ? out.begin() : out.begin() + 4;
        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(in);
          benchmark::DoNotOptimize(out);
          auto result = copy(first, last, dst);
          benchmark::DoNotOptimize(result);
        }
      })->Range(64, 1 << 20);
    };
    bm.operator()<true>("std::copy(vector<bool>) (aligned)", std_copy);
    bm.operator()<false>("std::copy(vector<bool>) (unaligned)", std_copy);
#if TEST_STD_VER >= 23 // vector<bool>::iterator is not an output_iterator before C++23
    bm.operator()<true>("rng::copy(vector<bool>) (aligned)", std::ranges::copy);
    bm.operator()<false>("rng::copy(vector<bool>) (unaligned)", std::ranges::copy);
#endif
  }

  // {std,ranges}::copy(forward_iterator, forward_iterator, vector<bool>)
  {
    auto bm = []<template <class> class Iter>(std::string name, auto copy) {
      benchmark::RegisterBenchmark(name, [copy](auto& st) {
        std::size_t const n = st.range(0);
        std::vector<int> in(n, 1);
        std::vector<bool> out(n);
        auto first = Iter(in.begin());
        auto last  = Iter(in.end());
        auto dst   = out.begin();
        for ([[maybe_unused]] auto _ : st) {
          benchmark::DoNotOptimize(in);
          benchmark::DoNotOptimize(out);
          auto result = copy(first, last, dst);
          benchmark::DoNotOptimize(result);
        }
      })->Range(64, 1 << 20);
    };
    bm.operator()<forward_iterator>("std::copy(forward_iterator, vector<bool>)", std_copy);
    bm.operator()<random_access_iterator>("std::copy(random_access_iterator, vector<bool>)", std_copy);
#if TEST_STD_VER >= 23 // vector<bool>::iterator is not an output_iterator before C++23
    bm.operator()<forward_iterator>("rng::copy(forward_iterator, vector<bool>)", std::ranges::copy);
    bm.operator()<random_access_iterator>("rng::copy(random_access_iterator, vector<bool>)", std::ranges::copy);
#endif
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
