//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <execution>
#include <utility>

#include <benchmark/benchmark.h>

#include "common.h"
#include "test_macros.h"

int main(int argc, char** argv) {
  auto bm = [](std::string name, auto&& policy, auto generate_data) {
    benchmark::RegisterBenchmark(
        name,
        [&policy, generate_data](auto& st) mutable {
          constexpr std::size_t BatchSize = 32;
          std::size_t size                = st.range(0);
          std::vector<int> data           = generate_data(size);
          std::array<std::vector<int>, BatchSize> c;
          std::fill_n(c.begin(), BatchSize, data);

          while (st.KeepRunningBatch(BatchSize)) {
            for (std::size_t i = 0; i != BatchSize; ++i) {
              benchmark::DoNotOptimize(c[i]);
              std::sort(policy, c[i].begin(), c[i].end());
              benchmark::DoNotOptimize(c[i]);
            }

            // Reset c to its original unsorted state
            st.PauseTiming();
            for (std::size_t i = 0; i != BatchSize; ++i) {
              std::copy(data.begin(), data.end(), c[i].begin());
            }
            st.ResumeTiming();
          }
        })
        ->Arg(1 << 6)  // 64
        ->Arg(1 << 16) // 65'536
        ->Arg(1 << 26) // 67'108'864
        ->UseRealTime();
  };

  auto register_bm = [&](auto generate, std::string variant) {
    auto name = [variant](std::string op) { return op + " (" + variant + ")"; };
#if defined(TEST_PSTL_ENABLE_SEQ_BASELINES)
    bm.operator()(name("std::sort(std::execution::seq, vector<int>)"), std::execution::seq, generate);
#endif
    bm.operator()(name("std::sort(std::execution::par, vector<int>)"), std::execution::par, generate);
  };

  register_bm(support::quicksort_adversarial_data<int>, "qsort adversarial");
  register_bm(support::ascending_sorted_data<int>, "ascending");
  register_bm(support::descending_sorted_data<int>, "descending");
  register_bm(support::pipe_organ_data<int>, "pipe-organ");
  register_bm(support::heap_data<int>, "heap");
  register_bm(support::shuffled_data<int>, "shuffled");
  register_bm(support::single_element_data<int>, "repeated");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
