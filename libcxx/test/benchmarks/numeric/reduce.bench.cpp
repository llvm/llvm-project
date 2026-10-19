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

// Keep values small so that signed integer arithmetic does not overflow even for the largest inputs
template <class T>
T random_value() {
  if constexpr (std::is_integral_v<T>)
    return getRandomInteger<T>(-100, 100);
  else
    return getRandomFloat<T>(-100, 100);
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
            ValueType init = random_value<ValueType>();

            auto reduce = [](auto x, auto y) { return x + y; };

            for ([[maybe_unused]] auto _ : st) {
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
