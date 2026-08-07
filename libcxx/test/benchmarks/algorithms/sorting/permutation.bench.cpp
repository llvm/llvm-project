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
#include <list>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "common.h"

int main(int argc, char** argv) {
  auto std_next_permutation = [](auto first, auto last) { return std::next_permutation(first, last); };
  auto std_prev_permutation = [](auto first, auto last) { return std::prev_permutation(first, last); };

  // Benchmark next_permutation and prev_permutation. We walk the permutation sequence in place,
  // calling the algorithm repeatedly without restoring between calls: these algorithms cycle through
  // all permutations, so this measures the realistic amortized per-step cost.
  //
  // We intentionally don't benchmark the predicated overloads because that makes the benchmark
  // run too slowly.
  {
    auto bm = []<class Container>(std::string name, auto permutation, auto generate_data) {
      benchmark::RegisterBenchmark(
          name,
          [permutation, generate_data](auto& st) {
            std::size_t const size      = st.range(0);
            using ValueType             = typename Container::value_type;
            std::vector<ValueType> data = generate_data(size);
            Container c(data.begin(), data.end());

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              auto result = permutation(c.begin(), c.end());
              benchmark::DoNotOptimize(result);
              benchmark::DoNotOptimize(c);
            }
          })
          ->Arg(8)
          ->Arg(1024)
          ->Arg(8192);
    };

    auto gen_int         = [](std::size_t size) { return support::shuffled_data<int>(size); };
    auto gen_nonintegral = [](std::size_t size) {
      std::vector<int> data = support::shuffled_data<int>(size);
      return std::vector<support::NonIntegral>(data.begin(), data.end());
    };

    // clang-format off
    bm.operator()<std::vector<int>>("std::next_permutation(vector<int>)", std_next_permutation, gen_int);
    bm.operator()<std::vector<support::NonIntegral>>("std::next_permutation(vector<NonIntegral>)", std_next_permutation, gen_nonintegral);
    bm.operator()<std::deque<int>>("std::next_permutation(deque<int>)", std_next_permutation, gen_int);
    bm.operator()<std::list<int>>("std::next_permutation(list<int>)", std_next_permutation, gen_int);

    bm.operator()<std::vector<int>>("std::prev_permutation(vector<int>)", std_prev_permutation, gen_int);
    bm.operator()<std::vector<support::NonIntegral>>("std::prev_permutation(vector<NonIntegral>)", std_prev_permutation, gen_nonintegral);
    bm.operator()<std::deque<int>>("std::prev_permutation(deque<int>)", std_prev_permutation, gen_int);
    bm.operator()<std::list<int>>("std::prev_permutation(list<int>)", std_prev_permutation, gen_int);
    // clang-format on
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
