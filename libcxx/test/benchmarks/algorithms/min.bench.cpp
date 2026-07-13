//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cassert>
#include <deque>
#include <list>
#include <vector>

#include "test_macros.h"
#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  auto bm = []<class Container>(std::type_identity<Container>, std::string name) {
    benchmark::RegisterBenchmark(
        name,
        [](benchmark::State& state) {
          Container vec(state.range(), 3);

          for (auto _ : state) {
            benchmark::DoNotOptimize(vec);
            benchmark::DoNotOptimize(std::ranges::min(vec));
          }
        })
        ->Arg(1)
        ->Arg(8)
        ->Arg(64)
        ->Arg(70000);
  };

  bm(std::type_identity<std::vector<char>>(), "ranges::min(std::vector<char>)");
  bm(std::type_identity<std::vector<long long>>(), "ranges::min(std::vector<long long>)");
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::vector<__int128>>(), "ranges::min(std::vector<__int128>)");
#endif

  bm(std::type_identity<std::deque<char>>(), "ranges::min(std::deque<char>)");
  bm(std::type_identity<std::deque<long long>>(), "ranges::min(std::deque<long long>)");
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::deque<__int128>>(), "ranges::min(std::deque<__int128>)");
#endif

  bm(std::type_identity<std::list<char>>(), "ranges::min(std::list<char>)");
  bm(std::type_identity<std::list<long long>>(), "ranges::min(std::list<long long>)");
#ifndef TEST_HAS_NO_INT128
  bm(std::type_identity<std::list<__int128>>(), "ranges::min(std::list<__int128>)");
#endif

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
