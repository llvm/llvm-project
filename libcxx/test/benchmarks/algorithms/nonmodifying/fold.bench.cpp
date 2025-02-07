//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <limits>
#include <list>
#include <string>
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
            ValueType const limit  = 1000; // ensure we never overflow in the addition
            assert(size <= std::numeric_limits<ValueType>::max());
            assert(std::numeric_limits<ValueType>::max() > static_cast<ValueType>(size) * limit);
            assert(std::numeric_limits<ValueType>::min() < static_cast<ValueType>(size) * limit * -1);

            Container c;
            std::generate_n(std::back_inserter(c), size, [&] {
              return std::clamp(Generate<ValueType>::random(), -1 * limit, limit);
            });
            ValueType init = c.back();
            c.pop_back();

            auto f = [](auto x, auto y) {
              benchmark::DoNotOptimize(x);
              benchmark::DoNotOptimize(y);
              return x + y;
            };

            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(init);
              auto result = fold(c.begin(), c.end(), init, f);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(32)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm.operator()<std::vector<int>>("rng::fold_left(vector<int>)", std::ranges::fold_left);
    bm.operator()<std::deque<int>>("rng::fold_left(deque<int>)", std::ranges::fold_left);
    bm.operator()<std::list<int>>("rng::fold_left(list<int>)", std::ranges::fold_left);

    // fold_right not implemented yet
    // bm.operator()<std::vector<int>>("rng::fold_right(vector<int>)", std::ranges::fold_right);
    // bm.operator()<std::deque<int>>("rng::fold_right(deque<int>)", std::ranges::fold_right);
    // bm.operator()<std::list<int>>("rng::fold_right(list<int>)", std::ranges::fold_right);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
