//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <list>
#include <string>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  // ranges::{fold_left,fold_right}
  {
    auto bm = []<class Container>(std::string name, auto fold) {
      benchmark::RegisterBenchmark(
          name,
          [fold](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            static_assert(std::is_unsigned_v<ValueType>,
                          "We could encounter UB if signed arithmetic overflows in this benchmark");

            Container c;
            std::generate_n(std::back_inserter(c), size, [&] { return Generate<ValueType>::random(); });
            ValueType init = c.back();
            c.pop_back();

            auto f = [](auto x, auto y) {
              benchmark::DoNotOptimize(x);
              benchmark::DoNotOptimize(y);
              return x + y;
            };

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(init);
              auto result = fold(c.begin(), c.end(), init, f);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<unsigned int>>("rng::fold_left(vector<int>)", std::ranges::fold_left);
    bm.operator()<std::deque<unsigned int>>("rng::fold_left(deque<int>)", std::ranges::fold_left);
    bm.operator()<std::list<unsigned int>>("rng::fold_left(list<int>)", std::ranges::fold_left);

    // TODO: fold_right not implemented yet
    // bm.operator()<std::vector<unsigned int>>("rng::fold_right(vector<int>)", std::ranges::fold_right);
    // bm.operator()<std::deque<unsigned int>>("rng::fold_right(deque<int>)", std::ranges::fold_right);
    // bm.operator()<std::list<unsigned int>>("rng::fold_right(list<int>)", std::ranges::fold_right);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
