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
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <execution>
#include <utility>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto bm = [](std::string name, auto&& policy, bool has_needle) {
    benchmark::RegisterBenchmark(
        name,
        [&policy, has_needle](auto& st) mutable {
          std::size_t size = st.range(0);
          double x         = Generate<double>::random();
          double y         = random_different_from({x});
          std::vector<double> c(size, x);

          if (has_needle) {
            // put the element we're searching for at 25% of the sequence
            *std::next(c.begin(), size / 4) = y;
          }

          for ([[maybe_unused]] auto _ : st) {
            benchmark::DoNotOptimize(c);
            benchmark::DoNotOptimize(y);
            auto result = std::find(policy, c.begin(), c.end(), y);
            benchmark::DoNotOptimize(result);
          }
        })
        ->Arg(1 << 6)  // 64
        ->Arg(1 << 16) // 65'536
        ->Arg(1 << 26) // 67'108'864
        ->UseRealTime();
  };
#if defined(TEST_PSTL_ENABLE_SEQ_BASELINES)
  bm("std::find(std::execution::seq, vector<double>) (bail 25%)", std::execution::seq, true);
  bm("std::find(std::execution::seq, vector<double>) (process all)", std::execution::seq, false);
#endif
  bm("std::find(std::execution::par, vector<double>) (bail 25%)", std::execution::par, true);
  bm("std::find(std::execution::par, vector<double>) (process all)", std::execution::par, false);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
