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
#include <list>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

template <class Container, class Operation>
void bm_general(std::string operation_name, Operation copy_n) {
  auto bench = [copy_n](auto& st) {
    auto const size = st.range(0);
    using ValueType = typename Container::value_type;
    Container c;
    std::generate_n(std::back_inserter(c), size, [] { return Generate<ValueType>::random(); });

    std::vector<ValueType> out(size);

    for ([[maybe_unused]] auto _ : st) {
      auto result = copy_n(c.begin(), size, out.begin());
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(out);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Range(8, 1 << 20);
}

template <bool Aligned, class Operation>
static void bm_vector_bool(std::string operation_name, Operation copy_n) {
  auto bench = [copy_n](auto& st) {
    auto n = st.range();
    std::vector<bool> in(n, true);
    std::vector<bool> out(Aligned ? n : n + 8);
    benchmark::DoNotOptimize(&in);
    auto first = in.begin();
    auto dst   = Aligned ? out.begin() : out.begin() + 4;
    for ([[maybe_unused]] auto _ : st) {
      auto result = copy_n(first, n, dst);
      benchmark::DoNotOptimize(result);
      benchmark::DoNotOptimize(out);
      benchmark::ClobberMemory();
    }
  };
  benchmark::RegisterBenchmark(operation_name, bench)->Range(64, 1 << 20);
}

int main(int argc, char** argv) {
  auto std_copy_n    = [](auto first, auto n, auto out) { return std::copy_n(first, n, out); };
  auto ranges_copy_n = [](auto first, auto n, auto out) { return std::ranges::copy_n(first, n, out); };

  // std::copy_n
  bm_general<std::vector<int>>("std::copy_n(vector<int>)", std_copy_n);
  bm_general<std::deque<int>>("std::copy_n(deque<int>)", std_copy_n);
  bm_general<std::list<int>>("std::copy_n(list<int>)", std_copy_n);
  bm_vector_bool<true>("std::copy_n(vector<bool>) (aligned)", std_copy_n);
  bm_vector_bool<false>("std::copy_n(vector<bool>) (unaligned)", std_copy_n);

  // ranges::copy_n
  bm_general<std::vector<int>>("ranges::copy_n(vector<int>)", ranges_copy_n);
  bm_general<std::deque<int>>("ranges::copy_n(deque<int>)", ranges_copy_n);
  bm_general<std::list<int>>("ranges::copy_n(list<int>)", ranges_copy_n);
  bm_vector_bool<true>("ranges::copy_n(vector<bool>) (aligned)", ranges_copy_n);
  bm_vector_bool<false>("ranges::copy_n(vector<bool>) (unaligned)", ranges_copy_n);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
