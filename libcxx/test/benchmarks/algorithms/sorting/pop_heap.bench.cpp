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
#include <iterator>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto bm = []<class Container>(std::type_identity<Container>, std::string name) {
    benchmark::RegisterBenchmark(
        name,
        [](benchmark::State& state) {
          std::size_t size = state.range(0);

          Container c;
          std::generate_n(std::back_inserter(c), size, [] {
            return Generate<typename Container::value_type>::random();
          });

          while (state.KeepRunningBatch(size)) {
            state.PauseTiming();
            std::make_heap(c.begin(), c.end());
            state.ResumeTiming();

            for (auto first = c.begin(), last = c.end(); last != first; --last) {
              std::pop_heap(first, last);
            }
          }
        })
        ->Arg(8)
        ->Arg(1024)
        ->Arg(8192);
  };

  bm(std::type_identity<std::vector<int>>{}, "std::pop_heap(vector<int>)");
  bm(std::type_identity<std::vector<float>>{}, "std::pop_heap(vector<float>)");
  bm(std::type_identity<std::vector<size_t>>{}, "std::pop_heap(vector<size_t>)");
  bm(std::type_identity<std::vector<std::string>>{}, "std::pop_heap(vector<std::string>)");
  bm(std::type_identity<std::deque<int>>{}, "std::pop_heap(deque<int>)");
  bm(std::type_identity<std::deque<float>>{}, "std::pop_heap(deque<float>)");
  bm(std::type_identity<std::deque<size_t>>{}, "std::pop_heap(deque<size_t>)");
  bm(std::type_identity<std::deque<std::string>>{}, "std::pop_heap(deque<std::string>)");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
