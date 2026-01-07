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
#include <list>
#include <ranges>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  auto std_find    = [](auto first, auto last, auto const& value) { return std::find(first, last, value); };
  auto std_find_if = [](auto first, auto last, auto const& value) {
    return std::find_if(first, last, [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element == value;
    });
  };
  auto std_find_if_not = [](auto first, auto last, auto const& value) {
    return std::find_if_not(first, last, [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element != value;
    });
  };

  auto ranges_find    = [](auto first, auto last, auto const& value) { return std::ranges::find(first, last, value); };
  auto ranges_find_if = [](auto first, auto last, auto const& value) {
    return std::ranges::find_if(first, last, [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element == value;
    });
  };
  auto ranges_find_if_not = [](auto first, auto last, auto const& value) {
    return std::ranges::find_if_not(first, last, [&](auto element) {
      benchmark::DoNotOptimize(element);
      return element != value;
    });
  };

  auto register_benchmarks = [&](auto bm, std::string comment) {
    // find
    bm.template operator()<std::vector<char>>("std::find(vector<char>) (" + comment + ")", std_find);
    bm.template operator()<std::vector<short>>("std::find(vector<short>) (" + comment + ")", std_find);
    bm.template operator()<std::vector<int>>("std::find(vector<int>) (" + comment + ")", std_find);
    bm.template operator()<std::vector<long long>>("std::find(vector<long long>) (" + comment + ")", std_find);
    bm.template operator()<std::deque<int>>("std::find(deque<int>) (" + comment + ")", std_find);
    bm.template operator()<std::list<int>>("std::find(list<int>) (" + comment + ")", std_find);

    bm.template operator()<std::vector<char>>("rng::find(vector<char>) (" + comment + ")", ranges_find);
    bm.template operator()<std::vector<int>>("rng::find(vector<int>) (" + comment + ")", ranges_find);
    bm.template operator()<std::deque<int>>("rng::find(deque<int>) (" + comment + ")", ranges_find);
    bm.template operator()<std::list<int>>("rng::find(list<int>) (" + comment + ")", ranges_find);

    // find_if
    bm.template operator()<std::vector<char>>("std::find_if(vector<char>) (" + comment + ")", std_find_if);
    bm.template operator()<std::vector<int>>("std::find_if(vector<int>) (" + comment + ")", std_find_if);
    bm.template operator()<std::deque<int>>("std::find_if(deque<int>) (" + comment + ")", std_find_if);
    bm.template operator()<std::list<int>>("std::find_if(list<int>) (" + comment + ")", std_find_if);

    bm.template operator()<std::vector<char>>("rng::find_if(vector<char>) (" + comment + ")", ranges_find_if);
    bm.template operator()<std::vector<int>>("rng::find_if(vector<int>) (" + comment + ")", ranges_find_if);
    bm.template operator()<std::deque<int>>("rng::find_if(deque<int>) (" + comment + ")", ranges_find_if);
    bm.template operator()<std::list<int>>("rng::find_if(list<int>) (" + comment + ")", ranges_find_if);

    // find_if_not
    bm.template operator()<std::vector<char>>("std::find_if_not(vector<char>) (" + comment + ")", std_find_if_not);
    bm.template operator()<std::vector<int>>("std::find_if_not(vector<int>) (" + comment + ")", std_find_if_not);
    bm.template operator()<std::deque<int>>("std::find_if_not(deque<int>) (" + comment + ")", std_find_if_not);
    bm.template operator()<std::list<int>>("std::find_if_not(list<int>) (" + comment + ")", std_find_if_not);

    bm.template operator()<std::vector<char>>("rng::find_if_not(vector<char>) (" + comment + ")", ranges_find_if_not);
    bm.template operator()<std::vector<int>>("rng::find_if_not(vector<int>) (" + comment + ")", ranges_find_if_not);
    bm.template operator()<std::deque<int>>("rng::find_if_not(deque<int>) (" + comment + ")", ranges_find_if_not);
    bm.template operator()<std::list<int>>("rng::find_if_not(list<int>) (" + comment + ")", ranges_find_if_not);
  };

  auto register_nested_container_benchmarks = [&](auto bm, std::string comment) {
    // ranges_find
    bm.template operator()<std::vector<std::vector<char>>>(
        "rng::find(join_view(vector<vector<char>>)) (" + comment + ")", ranges_find);
    bm.template operator()<std::vector<std::vector<int>>>(
        "rng::find(join_view(vector<vector<int>>)) (" + comment + ")", ranges_find);
    bm.template operator()<std::list<std::vector<int>>>(
        "rng::find(join_view(list<vector<int>>)) (" + comment + ")", ranges_find);
    bm.template operator()<std::vector<std::list<int>>>(
        "rng::find(join_view(vector<list<int>>)) (" + comment + ")", ranges_find);
    bm.template operator()<std::deque<std::deque<int>>>(
        "rng::find(join_view(deque<deque<int>>)) (" + comment + ")", ranges_find);
  };

  // Benchmark {std,ranges}::{find,find_if,find_if_not}(normal container) where we
  // bail out after 25% of elements
  {
    auto bm = []<class Container>(std::string name, auto find) {
      benchmark::RegisterBenchmark(
          name,
          [find](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);

            // put the element we're searching for at 25% of the sequence
            *std::next(c.begin(), size / 4) = y;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(y);
              auto result = find(c.begin(), c.end(), y);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 15);
    };
    register_benchmarks(bm, "bail 25%");
  }

  // Benchmark {std,ranges}::{find,find_if,find_if_not}(normal container) where we process the whole sequence
  {
    auto bm = []<class Container>(std::string name, auto find) {
      benchmark::RegisterBenchmark(
          name,
          [find](auto& st) {
            std::size_t const size = st.range(0);
            using ValueType        = typename Container::value_type;
            ValueType x            = Generate<ValueType>::random();
            ValueType y            = random_different_from({x});
            Container c(size, x);

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(y);
              auto result = find(c.begin(), c.end(), y);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 15);
    };
    register_benchmarks(bm, "process all");
  }

  // Benchmark {std,ranges}::{find,find_if,find_if_not}(join(normal container)) where we process the whole sequence
  {
    auto bm = []<class Container>(std::string name, auto find) {
      benchmark::RegisterBenchmark(
          name,
          [find](auto& st) {
            std::size_t const size     = st.range(0);
            std::size_t const seg_size = 256;
            std::size_t const segments = (size + seg_size - 1) / seg_size;
            using C1                   = typename Container::value_type;
            using ValueType            = typename C1::value_type;
            ValueType x                = Generate<ValueType>::random();
            ValueType y                = random_different_from({x});
            Container c(segments);
            auto n = size;
            for (auto it = c.begin(); it != c.end(); it++) {
              it->resize(std::min(seg_size, n), x);
              n -= it->size();
            }

            auto view = c | std::views::join;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(y);
              auto result = find(view.begin(), view.end(), y);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 15);
    };
    register_nested_container_benchmarks(bm, "process all");
  }

  // Benchmark {std,ranges}::{find,find_if,find_if_not}(vector<bool>) where we process the whole sequence
  {
    auto bm = [](std::string name, auto find) {
      benchmark::RegisterBenchmark(
          name,
          [find](auto& st) {
            std::size_t const size = st.range(0);
            std::vector<bool> c(size, true);
            bool y = false;

            for ([[maybe_unused]] auto _ : st) {
              benchmark::DoNotOptimize(c);
              benchmark::DoNotOptimize(y);
              auto result = find(c.begin(), c.end(), y);
              benchmark::DoNotOptimize(result);
            }
          })
          ->Arg(8)
          ->Arg(50) // non power-of-two
          ->Arg(1024)
          ->Arg(8192)
          ->Arg(1 << 20);
    };
    bm("std::find(vector<bool>) (process all)", std_find);
    bm("rng::find(vector<bool>) (process all)", ranges_find);

    bm("std::find_if(vector<bool>) (process all)", std_find_if);
    bm("rng::find_if(vector<bool>) (process all)", ranges_find_if);

    bm("std::find_if_not(vector<bool>) (process all)", std_find_if_not);
    bm("rng::find_if_not(vector<bool>) (process all)", ranges_find_if_not);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
