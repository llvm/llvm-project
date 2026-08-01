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
#include <forward_list>
#include <list>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_binary_search = [](auto first, auto last, auto const& value) {
    return std::binary_search(first, last, value);
  };
  auto std_binary_search_pred = [](auto first, auto last, auto const& value) {
    return std::binary_search(first, last, value, [](auto x, auto y) { return x < y; });
  };

  // Benchmark binary_search looking up a random key that is present in the sequence.
  {
    auto bm = []<class Container>(std::string name, auto binary_search) {
      benchmark::RegisterBenchmark(
          name,
          [binary_search](auto& st) {
            using ValueType        = typename Container::value_type;
            std::size_t const size = st.range(0);

            // Random sorted data
            std::vector<ValueType> data(size);
            std::generate_n(data.begin(), size, &Generate<ValueType>::random);
            std::sort(data.begin(), data.end());

            // Precompute a bunch of random keys.
            std::vector<ValueType> keys(data);
            std::shuffle(keys.begin(), keys.end(), getRandomEngine());

            Container c(data.begin(), data.end());
            std::size_t pos = 0;
            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              bool result = binary_search(c.begin(), c.end(), keys[pos]);
              benchmark::DoNotOptimize(result);
              if (++pos == keys.size())
                pos = 0;
            }
          })
          ->Arg(8)
          ->Arg(100)
          ->Arg(8192);
    };
    // clang-format off
    bm.operator()<std::vector<int>>("std::binary_search(vector<int>) (present)", std_binary_search);
    bm.operator()<std::deque<int>>("std::binary_search(deque<int>) (present)", std_binary_search);
    bm.operator()<std::list<int>>("std::binary_search(list<int>) (present)", std_binary_search);
    bm.operator()<std::forward_list<int>>("std::binary_search(forward_list<int>) (present)", std_binary_search);

    bm.operator()<std::vector<int>>("std::binary_search(vector<int>, pred) (present)", std_binary_search_pred);
    bm.operator()<std::deque<int>>("std::binary_search(deque<int>, pred) (present)", std_binary_search_pred);
    bm.operator()<std::list<int>>("std::binary_search(list<int>, pred) (present)", std_binary_search_pred);
    bm.operator()<std::forward_list<int>>("std::binary_search(forward_list<int>, pred) (present)", std_binary_search_pred);
    // clang-format on
  }

  // Benchmark binary_search looking up a key that is absent from the sequence.
  {
    auto bm = []<class Container>(std::string name, auto binary_search) {
      benchmark::RegisterBenchmark(
          name,
          [binary_search](auto& st) {
            using ValueType        = typename Container::value_type;
            std::size_t const size = st.range(0);

            // Random sorted data
            std::vector<ValueType> data(size);
            std::generate_n(data.begin(), size, &Generate<ValueType>::random);
            std::sort(data.begin(), data.end());

            // Find a key that isn't in the sequence
            ValueType absent = Generate<ValueType>::random();
            while (std::find(data.begin(), data.end(), absent) != data.end())
              absent = Generate<ValueType>::random();

            Container c(data.begin(), data.end());
            for (auto _ : st) {
              benchmark::DoNotOptimize(c);
              bool result = binary_search(c.begin(), c.end(), absent);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(100)
          ->Arg(8192);
    };
    // clang-format off
    bm.operator()<std::vector<int>>("std::binary_search(vector<int>) (absent)", std_binary_search);
    bm.operator()<std::deque<int>>("std::binary_search(deque<int>) (absent)", std_binary_search);
    bm.operator()<std::list<int>>("std::binary_search(list<int>) (absent)", std_binary_search);
    bm.operator()<std::forward_list<int>>("std::binary_search(forward_list<int>) (absent)", std_binary_search);

    bm.operator()<std::vector<int>>("std::binary_search(vector<int>, pred) (absent)", std_binary_search_pred);
    bm.operator()<std::deque<int>>("std::binary_search(deque<int>, pred) (absent)", std_binary_search_pred);
    bm.operator()<std::list<int>>("std::binary_search(list<int>, pred) (absent)", std_binary_search_pred);
    bm.operator()<std::forward_list<int>>("std::binary_search(forward_list<int>, pred) (absent)", std_binary_search_pred);
    // clang-format on
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
