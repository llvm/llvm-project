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
  // A transformation/reduction that does just a bit more than a no-op
  struct minimal {
    double operator()(double x) const { return -x; }
    double operator()(double x, double y) const { return x + y; }
  };

  // A transformation/reduction that does miniscule work per element
  struct cheap {
    double operator()(double x) const { return -x + 0.25; }
    double operator()(double x, double y) const {
      // (compilers aren't allowed to optimize away the FP divisions and multiplications without fast-math)
      return x / 2. * 2. / 2. * 2. + y / 2. * 2. / 2. * 2.;
    }
  };

  // A transformation/reduction that does significant work per element
  struct expensive {
    double operator()(double x) const { return (-x + 0.25) * (x + 10.); }
    double operator()(double x, double y) const {
      // (compilers aren't allowed to optimize away the FP divisions and multiplications without fast-math)
      return x / 2. * 2. / 3. * 3. / 4. * 4. / 5. * 5. + y / 2. * 2. / 2. * 2. / 4. * 4. / 5. * 5.;
    }
  };

  auto bm = [](std::string name, auto&& policy, auto func) {
    benchmark::RegisterBenchmark(
        name,
        [&policy, func](auto& st) {
          std::size_t size = st.range(0);
          std::vector<double> c(size);
          std::iota(c.begin(), c.end(), 1.);
          for ([[maybe_unused]] auto _ : st) {
            benchmark::ClobberMemory();
            auto result = std::transform_reduce(policy, c.begin(), c.end(), 0.0, func, func);
            benchmark::DoNotOptimize(result);
          }
        })
        ->Arg(1 << 6)  // 64
        ->Arg(1 << 16) // 65'536
        ->Arg(1 << 26) // 67'108'864
        ->UseRealTime();
  };
#if defined(TEST_PSTL_ENABLE_SEQ_BASELINES)
  bm("std::transform_reduce(std::execution::seq, vector<double>) (minimal)", std::execution::seq, minimal{});
  bm("std::transform_reduce(std::execution::seq, vector<double>) (cheap)", std::execution::seq, cheap{});
  bm("std::transform_reduce(std::execution::seq, vector<double>) (expensive)", std::execution::seq, expensive{});
#endif
  bm("std::transform_reduce(std::execution::par, vector<double>) (minimal)", std::execution::par, minimal{});
  bm("std::transform_reduce(std::execution::par, vector<double>) (cheap)", std::execution::par, cheap{});
  bm("std::transform_reduce(std::execution::par, vector<double>) (expensive)", std::execution::par, expensive{});

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
