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
  // std::transform_reduce(first, last, init, reduce, transform)
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

            auto transform = [](auto x) { return x * 3; };
            auto reduce    = [](auto x, auto y) { return x + y; };

            for ([[maybe_unused]] auto _ : st) {
              benchmark::ClobberMemory();
              auto result = std::transform_reduce(c.begin(), c.end(), init, reduce, transform);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::transform_reduce(vector<int>)");
    bm.operator()<std::vector<float>>("std::transform_reduce(vector<float>)");
    bm.operator()<std::vector<double>>("std::transform_reduce(vector<double>)");
    bm.operator()<std::list<int>>("std::transform_reduce(list<int>)");
    bm.operator()<std::list<float>>("std::transform_reduce(list<float>)");
    bm.operator()<std::list<double>>("std::transform_reduce(list<double>)");
  }

  // std::transform_reduce(first1, last1, first2, init, reduce, transform)
  {
    auto bm = []<class Container>(std::string name) {
      benchmark::RegisterBenchmark(
          name,
          [](auto& st) TEST_ALIGN_BENCHMARK {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;

            Container c1, c2;
            std::generate_n(std::back_inserter(c1), size, [] { return random_value<ValueType>(); });
            std::generate_n(std::back_inserter(c2), size, [] { return random_value<ValueType>(); });
            ValueType init = random_value<ValueType>();

            auto transform = [](auto x, auto y) { return x - y; };
            auto reduce    = [](auto x, auto y) { return x + y; };

            for ([[maybe_unused]] auto _ : st) {
              benchmark::ClobberMemory();
              auto result = std::transform_reduce(c1.begin(), c1.end(), c2.begin(), init, reduce, transform);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    bm.operator()<std::vector<int>>("std::transform_reduce(vector<int>, vector<int>)");
    bm.operator()<std::vector<float>>("std::transform_reduce(vector<float>, vector<float>)");
    bm.operator()<std::vector<double>>("std::transform_reduce(vector<double>, vector<double>)");
    bm.operator()<std::list<int>>("std::transform_reduce(list<int>, list<int>)");
    bm.operator()<std::list<float>>("std::transform_reduce(list<float>, list<float>)");
    bm.operator()<std::list<double>>("std::transform_reduce(list<double>, list<double>)");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
