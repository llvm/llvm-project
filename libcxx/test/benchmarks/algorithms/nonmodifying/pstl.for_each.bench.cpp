//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>
#include <execution>
#include <utility>

#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  // A function that does just a bit more than a no-op
  auto minimal = [](double& x) { x = 0.; };

  // A function that does miniscule work per element
  auto cheap = [](double& x) { x = std::sin(x); };

  // A function that does significant work per element
  auto expensive = [](double& x) { x = std::pow(std::exp(std::sqrt(std::sin(std::cos(x)) + 1.0)), 1.7); };

  auto bm = [](std::string name, auto&& policy, auto&& func) {
    benchmark::RegisterBenchmark(
        name,
        [&policy, &func](auto& st) {
          std::size_t size = st.range(0);
          std::vector<double> c(size);
          std::iota(c.begin(), c.end(), 1.);
          for ([[maybe_unused]] auto _ : st) {
            benchmark::ClobberMemory();
            std::for_each(policy, c.begin(), c.end(), func);
            benchmark::DoNotOptimize(c);
          }
        })
        ->Arg(1 << 6)  // 64
        ->Arg(1 << 16) // 65'536
        ->Arg(1 << 26) // 67'108'864
        ->UseRealTime();
  };
#if defined(TEST_PSTL_ENABLE_SEQ_BASELINES)
  bm("std::for_each(std::execution::seq, vector<double>) (minimal)", std::execution::seq, minimal);
  bm("std::for_each(std::execution::seq, vector<double>) (cheap)", std::execution::seq, cheap);
  bm("std::for_each(std::execution::seq, vector<double>) (expensive)", std::execution::seq, expensive);
#endif
  bm("std::for_each(std::execution::par, vector<double>) (minimal)", std::execution::par, minimal);
  bm("std::for_each(std::execution::par, vector<double>) (cheap)", std::execution::par, cheap);
  bm("std::for_each(std::execution::par, vector<double>) (expensive)", std::execution::par, expensive);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
