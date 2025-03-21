//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <list>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_unique_copy      = [](auto first, auto last, auto out) { return std::unique_copy(first, last, out); };
  auto std_unique_copy_pred = [](auto first, auto last, auto out) {
    return std::unique_copy(first, last, out, [](auto a, auto b) {
      benchmark::DoNotOptimize(a);
      benchmark::DoNotOptimize(b);
      return a == b;
    });
  };
  auto ranges_unique_copy_pred = [](auto first, auto last, auto out) {
    return std::ranges::unique_copy(first, last, out, [](auto a, auto b) {
      benchmark::DoNotOptimize(a);
      benchmark::DoNotOptimize(b);
      return a == b;
    });
  };

  // Create a sequence of the form xxxxxxxxxxyyyyyyyyyy and unique the
  // adjacent equal elements.
  {
    auto bm = []<class Container>(std::string name, auto unique_copy) {
      benchmark::RegisterBenchmark(
          name,
          [unique_copy](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c(size);
            ValueType x = Generate<ValueType>::random();
            ValueType y = random_different_from({x});
            auto half   = size / 2;
            std::fill_n(std::fill_n(c.begin(), half, x), half, y);

            std::vector<ValueType> out(size);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(out);
              auto result = unique_copy(c.begin(), c.end(), out.begin());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::unique_copy(it, it, out)
    bm.operator()<std::vector<int>>("std::unique_copy(vector<int>) (contiguous)", std_unique_copy);
    bm.operator()<std::deque<int>>("std::unique_copy(deque<int>) (contiguous)", std_unique_copy);
    bm.operator()<std::list<int>>("std::unique_copy(list<int>) (contiguous)", std_unique_copy);
    bm.operator()<std::vector<int>>("rng::unique_copy(vector<int>) (contiguous)", std::ranges::unique_copy);
    bm.operator()<std::deque<int>>("rng::unique_copy(deque<int>) (contiguous)", std::ranges::unique_copy);
    bm.operator()<std::list<int>>("rng::unique_copy(list<int>) (contiguous)", std::ranges::unique_copy);

    // {std,ranges}::unique_copy(it, it, out, pred)
    bm.operator()<std::vector<int>>("std::unique_copy(vector<int>, pred) (contiguous)", std_unique_copy_pred);
    bm.operator()<std::deque<int>>("std::unique_copy(deque<int>, pred) (contiguous)", std_unique_copy_pred);
    bm.operator()<std::list<int>>("std::unique_copy(list<int>, pred) (contiguous)", std_unique_copy_pred);
    bm.operator()<std::vector<int>>("rng::unique_copy(vector<int>, pred) (contiguous)", ranges_unique_copy_pred);
    bm.operator()<std::deque<int>>("rng::unique_copy(deque<int>, pred) (contiguous)", ranges_unique_copy_pred);
    bm.operator()<std::list<int>>("rng::unique_copy(list<int>, pred) (contiguous)", ranges_unique_copy_pred);
  }

  // Create a sequence of the form xxyyxxyyxxyyxxyyxxyy and unique
  // adjacent equal elements.
  {
    auto bm = []<class Container>(std::string name, auto unique_copy) {
      benchmark::RegisterBenchmark(
          name,
          [unique_copy](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            Container c(size);
            ValueType x   = Generate<ValueType>::random();
            ValueType y   = random_different_from({x});
            auto populate = [&](Container& cont) {
              assert(cont.size() % 4 == 0);
              auto out = cont.begin();
              for (std::size_t i = 0; i != cont.size(); i += 4) {
                *out++ = x;
                *out++ = x;
                *out++ = y;
                *out++ = y;
              }
            };
            populate(c);

            std::vector<ValueType> out(size);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(out);
              auto result = unique_copy(c.begin(), c.end(), out.begin());
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(32)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192);
    };
    // {std,ranges}::unique_copy(it, it, out)
    bm.operator()<std::vector<int>>("std::unique_copy(vector<int>) (sprinkled)", std_unique_copy);
    bm.operator()<std::deque<int>>("std::unique_copy(deque<int>) (sprinkled)", std_unique_copy);
    bm.operator()<std::list<int>>("std::unique_copy(list<int>) (sprinkled)", std_unique_copy);
    bm.operator()<std::vector<int>>("rng::unique_copy(vector<int>) (sprinkled)", std::ranges::unique_copy);
    bm.operator()<std::deque<int>>("rng::unique_copy(deque<int>) (sprinkled)", std::ranges::unique_copy);
    bm.operator()<std::list<int>>("rng::unique_copy(list<int>) (sprinkled)", std::ranges::unique_copy);

    // {std,ranges}::unique_copy(it, it, out, pred)
    bm.operator()<std::vector<int>>("std::unique_copy(vector<int>, pred) (sprinkled)", std_unique_copy_pred);
    bm.operator()<std::deque<int>>("std::unique_copy(deque<int>, pred) (sprinkled)", std_unique_copy_pred);
    bm.operator()<std::list<int>>("std::unique_copy(list<int>, pred) (sprinkled)", std_unique_copy_pred);
    bm.operator()<std::vector<int>>("rng::unique_copy(vector<int>, pred) (sprinkled)", ranges_unique_copy_pred);
    bm.operator()<std::deque<int>>("rng::unique_copy(deque<int>, pred) (sprinkled)", ranges_unique_copy_pred);
    bm.operator()<std::list<int>>("rng::unique_copy(list<int>, pred) (sprinkled)", ranges_unique_copy_pred);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
