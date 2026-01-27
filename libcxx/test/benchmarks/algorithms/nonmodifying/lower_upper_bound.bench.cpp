//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <algorithm>
#include <deque>
#include <forward_list>
#include <list>
#include <numeric>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "../../GenerateInput.h"

int main(int argc, char** argv) {
  {
    auto bm = []<class Container>(benchmark::State& state) static {
      std::mt19937_64 rng{123456};

      using value_type = typename Container::value_type;

      std::vector<value_type> vec;
      for (int64_t i = 0; i != state.range(); ++i)
        vec.emplace_back(Generate<typename Container::value_type>::random());
      std::sort(vec.begin(), vec.end());

      Container c(vec.begin(), vec.end());

      for (auto _ : state) {
        auto result = std::lower_bound(c.begin(), c.end(), vec[rng() % vec.size()]);
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(vec);
      }
    };

    auto register_benchmark = [&]<class Container>(std::type_identity<Container>, std::string name) {
      benchmark::RegisterBenchmark(name, bm.template operator()<Container>)->Arg(8)->Arg(100)->Arg(8192);
    };

    register_benchmark(std::type_identity<std::vector<int>>{}, "std::lower_bound(std::vector<int>)");
    register_benchmark(std::type_identity<std::deque<int>>{}, "std::lower_bound(std::deque<int>)");
    register_benchmark(std::type_identity<std::list<int>>{}, "std::lower_bound(std::list<int>)");
    register_benchmark(std::type_identity<std::forward_list<int>>{}, "std::lower_bound(std::forward_list<int>)");
  }
  {
    auto bm = []<class Container>(benchmark::State& state) static {
      std::mt19937_64 rng{123456};

      using value_type = typename Container::value_type;

      std::vector<value_type> vec;
      for (int64_t i = 0; i != state.range(); ++i)
        vec.emplace_back(Generate<typename Container::value_type>::random());
      std::sort(vec.begin(), vec.end());

      Container c(vec.begin(), vec.end());

      for (auto _ : state) {
        auto result = std::upper_bound(c.begin(), c.end(), vec[rng() % vec.size()]);
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(vec);
      }
    };

    auto register_benchmark = [&]<class Container>(std::type_identity<Container>, std::string name) {
      benchmark::RegisterBenchmark(name, bm.template operator()<Container>)->Arg(8)->Arg(100)->Arg(8192);
    };

    register_benchmark(std::type_identity<std::vector<int>>{}, "std::upper_bound(std::vector<int>)");
    register_benchmark(std::type_identity<std::deque<int>>{}, "std::upper_bound(std::deque<int>)");
    register_benchmark(std::type_identity<std::list<int>>{}, "std::upper_bound(std::list<int>)");
    register_benchmark(std::type_identity<std::forward_list<int>>{}, "std::upper_bound(std::forward_list<int>)");
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
