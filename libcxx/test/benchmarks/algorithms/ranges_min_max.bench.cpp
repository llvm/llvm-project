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
        [algo](benchmark::State& state) {
          Container c(state.range(), 3);

          for (auto _ : state) {
            benchmark::DoNotOptimize(c);
            auto result = algo(c);
            benchmark::DoNotOptimize(result);
          }
        })
        ->Arg(1)
        ->Arg(8)
        ->Arg(64)
        ->Arg(70000);
  };

  // std::ranges::min
  bm(std::type_identity<std::vector<char>>(), "rng::min(std::vector<char>)", std::ranges::min);
  bm(std::type_identity<std::vector<long long>>(), "rng::min(std::vector<long long>)", std::ranges::min);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::vector<__int128>>(), "rng::min(std::vector<__int128>)", std::ranges::min);
#endif
  bm(std::type_identity<std::deque<char>>(), "rng::min(std::deque<char>)", std::ranges::min);
  bm(std::type_identity<std::deque<long long>>(), "rng::min(std::deque<long long>)", std::ranges::min);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::deque<__int128>>(), "rng::min(std::deque<__int128>)", std::ranges::min);
#endif
  bm(std::type_identity<std::list<char>>(), "rng::min(std::list<char>)", std::ranges::min);
  bm(std::type_identity<std::list<long long>>(), "rng::min(std::list<long long>)", std::ranges::min);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::list<__int128>>(), "rng::min(std::list<__int128>)", std::ranges::min);
#endif

  // std::ranges::max
  bm(std::type_identity<std::vector<char>>(), "rng::max(std::vector<char>)", std::ranges::max);
  bm(std::type_identity<std::vector<long long>>(), "rng::max(std::vector<long long>)", std::ranges::max);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::vector<__int128>>(), "rng::max(std::vector<__int128>)", std::ranges::max);
#endif
  bm(std::type_identity<std::deque<char>>(), "rng::max(std::deque<char>)", std::ranges::max);
  bm(std::type_identity<std::deque<long long>>(), "rng::max(std::deque<long long>)", std::ranges::max);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::deque<__int128>>(), "rng::max(std::deque<__int128>)", std::ranges::max);
#endif
  bm(std::type_identity<std::list<char>>(), "rng::max(std::list<char>)", std::ranges::max);
  bm(std::type_identity<std::list<long long>>(), "rng::max(std::list<long long>)", std::ranges::max);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::list<__int128>>(), "rng::max(std::list<__int128>)", std::ranges::max);
#endif

  // std::ranges::minmax
  bm(std::type_identity<std::vector<char>>(), "rng::minmax(std::vector<char>)", std::ranges::minmax);
  bm(std::type_identity<std::vector<long long>>(), "rng::minmax(std::vector<long long>)", std::ranges::minmax);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::vector<__int128>>(), "rng::minmax(std::vector<__int128>)", std::ranges::minmax);
#endif
  bm(std::type_identity<std::deque<char>>(), "rng::minmax(std::deque<char>)", std::ranges::minmax);
  bm(std::type_identity<std::deque<long long>>(), "rng::minmax(std::deque<long long>)", std::ranges::minmax);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::deque<__int128>>(), "rng::minmax(std::deque<__int128>)", std::ranges::minmax);
#endif
  bm(std::type_identity<std::list<char>>(), "rng::minmax(std::list<char>)", std::ranges::minmax);
  bm(std::type_identity<std::list<long long>>(), "rng::minmax(std::list<long long>)", std::ranges::minmax);
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::list<__int128>>(), "rng::minmax(std::list<__int128>)", std::ranges::minmax);
#endif

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
