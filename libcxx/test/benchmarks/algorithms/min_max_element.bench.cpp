//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <deque>
#include <list>
#include <vector>

#include "test_macros.h"
#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  auto bm = []<class Container>(std::type_identity<Container>, std::string name, auto algo) {
    benchmark::RegisterBenchmark(
        name,
        [algo](benchmark::State& state) TEST_ALIGN_BENCHMARK {
          Container c(state.range(), 3);

          for (auto _ : state) {
            benchmark::DoNotOptimize(c);
            auto result = algo(c);
            benchmark::DoNotOptimize(result);
          }
        })
        ->Arg(8)
        ->Arg(1024)
        ->Arg(8192)
        ->Arg(1 << 20);
  };

  // std::min_element
  auto min_element = [](auto& c) { return std::min_element(c.begin(), c.end()); };
  bm(std::type_identity<std::vector<char>>(), "std::min_element(std::vector<char>)", min_element);
  bm(std::type_identity<std::vector<long long>>(), "std::min_element(std::vector<long long>)", min_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::vector<__int128>>(), "std::min_element(std::vector<__int128>)", min_element);
#endif
  bm(std::type_identity<std::deque<char>>(), "std::min_element(std::deque<char>)", min_element);
  bm(std::type_identity<std::deque<long long>>(), "std::min_element(std::deque<long long>)", min_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::deque<__int128>>(), "std::min_element(std::deque<__int128>)", min_element);
#endif
  bm(std::type_identity<std::list<char>>(), "std::min_element(std::list<char>)", min_element);
  bm(std::type_identity<std::list<long long>>(), "std::min_element(std::list<long long>)", min_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::list<__int128>>(), "std::min_element(std::list<__int128>)", min_element);
#endif

  // std::max_element
  auto max_element = [](auto& c) { return std::max_element(c.begin(), c.end()); };
  bm(std::type_identity<std::vector<char>>(), "std::max_element(std::vector<char>)", max_element);
  bm(std::type_identity<std::vector<long long>>(), "std::max_element(std::vector<long long>)", max_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::vector<__int128>>(), "std::max_element(std::vector<__int128>)", max_element);
#endif
  bm(std::type_identity<std::deque<char>>(), "std::max_element(std::deque<char>)", max_element);
  bm(std::type_identity<std::deque<long long>>(), "std::max_element(std::deque<long long>)", max_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::deque<__int128>>(), "std::max_element(std::deque<__int128>)", max_element);
#endif
  bm(std::type_identity<std::list<char>>(), "std::max_element(std::list<char>)", max_element);
  bm(std::type_identity<std::list<long long>>(), "std::max_element(std::list<long long>)", max_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::list<__int128>>(), "std::max_element(std::list<__int128>)", max_element);
#endif

  // std::minmax_element
  auto minmax_element = [](auto& c) { return std::minmax_element(c.begin(), c.end()); };
  bm(std::type_identity<std::vector<char>>(), "std::minmax_element(std::vector<char>)", minmax_element);
  bm(std::type_identity<std::vector<long long>>(), "std::minmax_element(std::vector<long long>)", minmax_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::vector<__int128>>(), "std::minmax_element(std::vector<__int128>)", minmax_element);
#endif
  bm(std::type_identity<std::deque<char>>(), "std::minmax_element(std::deque<char>)", minmax_element);
  bm(std::type_identity<std::deque<long long>>(), "std::minmax_element(std::deque<long long>)", minmax_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::deque<__int128>>(), "std::minmax_element(std::deque<__int128>)", minmax_element);
#endif
  bm(std::type_identity<std::list<char>>(), "std::minmax_element(std::list<char>)", minmax_element);
  bm(std::type_identity<std::list<long long>>(), "std::minmax_element(std::list<long long>)", minmax_element);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::list<__int128>>(), "std::minmax_element(std::list<__int128>)", minmax_element);
#endif

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
