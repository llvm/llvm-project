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

template <class Container, class Operation>
void bm(std::string operation_name, Operation generate_n) {
  auto bench = [generate_n](auto& st) {
    std::size_t const size = st.range(0);
    Container c(size);
    using ValueType = typename Container::value_type;
    ValueType x     = Generate<ValueType>::random();

    for ([[maybe_unused]] auto _ : st) {
      auto f = [&x] { return x; };
      generate_n(c.begin(), size, f);
      benchmark::DoNotOptimize(c);
      benchmark::DoNotOptimize(x);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Arg(32)->Arg(1024)->Arg(8192);
}

int main(int argc, char** argv) {
  auto std_generate_n    = [](auto out, auto n, auto f) { return std::generate_n(out, n, f); };
  auto ranges_generate_n = [](auto out, auto n, auto f) { return std::ranges::generate_n(out, n, f); };

  // std::generate_n
  bm<std::vector<int>>("std::generate_n(vector<int>)", std_generate_n);
  bm<std::deque<int>>("std::generate_n(deque<int>)", std_generate_n);
  bm<std::list<int>>("std::generate_n(list<int>)", std_generate_n);

  // ranges::generate_n
  bm<std::vector<int>>("ranges::generate_n(vector<int>)", ranges_generate_n);
  bm<std::deque<int>>("ranges::generate_n(deque<int>)", ranges_generate_n);
  bm<std::list<int>>("ranges::generate_n(list<int>)", ranges_generate_n);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
