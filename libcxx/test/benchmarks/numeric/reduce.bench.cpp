//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <list>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>
#include "../GenerateInput.h"
#include "test_macros.h"

// Keep values small so that signed integer multiplication doesn't overflow
template <class T>
T random_value() {
  if constexpr (std::is_integral_v<T>)
    return getRandomInteger<T>(1, 1000);
  else
    return getRandomFloat<T>(1, 1000);
}

int main(int argc, char** argv) {
  // std::reduce(first, last, init, reduce)
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) TEST_ALIGN_BENCHMARK {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;

            Container c;
            std::generate_n(std::back_inserter(c), size, [] { return random_value<ValueType>(); });
            ValueType init = static_cast<ValueType>(1);

            auto reduce = [](auto x, auto y) {
              if constexpr (std::is_floating_point_v<decltype(x)>) {
                // Compilers aren't allowed to optimize away the FP divisions and multiplications without fast-math.
                return x / 3. * 3. / 5. * 5. + y / 7. * 7. / 9. * 9.;
              } else {
                // p=46337 is the largest prime with '(p-1)^2 <= INT_MAX', so the chain of reductions never overflows.
                constexpr decltype(x) p = 46337;
                return x * y % p;
              }
            };
            for (auto _ : st) {
              benchmark::ClobberMemory();
              auto result = std::reduce(c.begin(), c.end(), init, reduce);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::reduce(vector<int>)");
    bm.operator()<std::vector<float>>("std::reduce(vector<float>)");
    bm.operator()<std::vector<double>>("std::reduce(vector<double>)");
    bm.operator()<std::list<int>>("std::reduce(list<int>)");
    bm.operator()<std::list<float>>("std::reduce(list<float>)");
    bm.operator()<std::list<double>>("std::reduce(list<double>)");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
