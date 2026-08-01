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

#include <benchmark/benchmark.h>

// A function that does enough work per element to justify parallelization.
static void f(double& x) { x = std::pow(std::exp(std::sqrt(std::sin(std::cos(x)) + 1.0)), 42.0); };

int main(int argc, char** argv) {
  auto bm = [](std::string name, auto&& policy) {
    benchmark::RegisterBenchmark(
        name,
        [&policy](auto& st) {
          std::size_t const size = st.range(0);
          std::vector<double> c(size, 42.);
          std::iota(c.begin(), c.end(), 1.);
          auto first = c.begin();
          auto last  = c.end();
          for ([[maybe_unused]] auto _ : st) {
            benchmark::DoNotOptimize(c);
            std::for_each(policy, first, last, f);
            benchmark::DoNotOptimize(c);
          }
        })
        ->Arg(1024)
        ->Arg(8192)
        ->Arg(65536)
        ->Arg(524288)
        ->Arg(4194304)
        ->Arg(33554432);
  };
  bm.operator()("std::for_each(std::execution::seq, vector<double>)", std::execution::seq);
  bm.operator()("std::for_each(std::execution::par, vector<double>)", std::execution::par);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
