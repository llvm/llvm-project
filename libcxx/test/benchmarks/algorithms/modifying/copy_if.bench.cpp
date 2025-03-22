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
  auto std_copy_if = [](auto first, auto last, auto out, auto pred) { return std::copy_if(first, last, out, pred); };

  // Benchmark {std,ranges}::copy_if where we copy one out of two element, in alternance.
  // This is basically the worst case for this algorithm, I don't think there are many
  // optimizations that can be applied in this case.
  {
    auto bm = []<class Container>(std::string name, auto copy_if) {
      benchmark::RegisterBenchmark(name, [copy_if](auto& st) {
        std::size_t const n = st.range(0);
        using ValueType     = typename Container::value_type;
        Container c;
        std::generate_n(std::back_inserter(c), n, [] { return Generate<ValueType>::random(); });

        std::vector<ValueType> out(n);

        for ([[maybe_unused]] auto _ : st) {
          bool do_copy = false;
          auto pred    = [&do_copy](auto& element) {
            benchmark::DoNotOptimize(element);
            do_copy = !do_copy;
            return do_copy;
          };
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(out);
          auto result = copy_if(c.begin(), c.end(), out.begin(), pred);
          benchmark::DoNotOptimize(result);
        }
      })->Range(8, 1 << 20);
    };
    bm.operator()<std::vector<int>>("std::copy_if(vector<int>) (every other)", std_copy_if);
    bm.operator()<std::deque<int>>("std::copy_if(deque<int>) (every other)", std_copy_if);
    bm.operator()<std::list<int>>("std::copy_if(list<int>) (every other)", std_copy_if);

    bm.operator()<std::vector<int>>("rng::copy_if(vector<int>) (every other)", std::ranges::copy_if);
    bm.operator()<std::deque<int>>("rng::copy_if(deque<int>) (every other)", std::ranges::copy_if);
    bm.operator()<std::list<int>>("rng::copy_if(list<int>) (every other)", std::ranges::copy_if);
  }

  // Benchmark {std,ranges}::copy_if where we copy the full range.
  // Copy the full range.
  {
    auto bm = []<class Container>(std::string name, auto copy_if) {
      benchmark::RegisterBenchmark(name, [copy_if](auto& st) {
        std::size_t const n = st.range(0);
        using ValueType     = typename Container::value_type;
        Container c;
        std::generate_n(std::back_inserter(c), n, [] { return Generate<ValueType>::random(); });

        std::vector<ValueType> out(n);

        for ([[maybe_unused]] auto _ : st) {
          auto pred = [](auto& element) {
            benchmark::DoNotOptimize(element);
            return true;
          };
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(out);
          auto result = copy_if(c.begin(), c.end(), out.begin(), pred);
          benchmark::DoNotOptimize(result);
        }
      })->Range(8, 1 << 20);
    };
    bm.operator()<std::vector<int>>("std::copy_if(vector<int>) (entire range)", std_copy_if);
    bm.operator()<std::deque<int>>("std::copy_if(deque<int>) (entire range)", std_copy_if);
    bm.operator()<std::list<int>>("std::copy_if(list<int>) (entire range)", std_copy_if);

    bm.operator()<std::vector<int>>("rng::copy_if(vector<int>) (entire range)", std::ranges::copy_if);
    bm.operator()<std::deque<int>>("rng::copy_if(deque<int>) (entire range)", std::ranges::copy_if);
    bm.operator()<std::list<int>>("rng::copy_if(list<int>) (entire range)", std::ranges::copy_if);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
