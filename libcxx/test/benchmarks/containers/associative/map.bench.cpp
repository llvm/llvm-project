//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <map>
#include <string>
#include <string_view>
#include <utility>

#include "associative_container_benchmarks.h"
#include "../../GenerateInput.h"
#include "benchmark/benchmark.h"

static void BM_map_find_string_literal(benchmark::State& state) {
  std::map<std::string, int> map;
  map.emplace("Something very very long to show a long string situation", 1);
  map.emplace("Something Else", 2);

  for (auto _ : state) {
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.find("Something very very long to show a long string situation"));
  }
}

BENCHMARK(BM_map_find_string_literal);

// Benchmark: find()/contains()/at() with string_view. Demonstrates the benefit
// of __is_transparently_comparable_v for basic_string_view: the optimized path
// avoids constructing a temporary std::string (and its potential heap allocation
// for keys beyond SSO).

static constexpr const char* kLongKey = "Something very very long to show a long string situation";

static std::map<std::string, int> make_test_map() {
  std::map<std::string, int> map;
  map.emplace(kLongKey, 1);
  map.emplace("Something Else", 2);
  return map;
}

static void BM_map_find_string_view(benchmark::State& state) {
  auto map            = make_test_map();
  std::string_view sv = kLongKey;

  for (auto _ : state) {
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.find(sv));
  }
}

BENCHMARK(BM_map_find_string_view);

static void BM_map_contains_string_view(benchmark::State& state) {
  auto map            = make_test_map();
  std::string_view sv = kLongKey;

  for (auto _ : state) {
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.contains(sv));
  }
}

BENCHMARK(BM_map_contains_string_view);

static void BM_map_at_string_view(benchmark::State& state) {
  auto map            = make_test_map();
  std::string_view sv = kLongKey;

  for (auto _ : state) {
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.at(sv));
  }
}

BENCHMARK(BM_map_at_string_view);

template <class K, class V>
struct support::adapt_operations<std::map<K, V>> {
  using ValueType = typename std::map<K, V>::value_type;
  using KeyType   = typename std::map<K, V>::key_type;
  static ValueType value_from_key(KeyType const& k) { return {k, Generate<V>::arbitrary()}; }
  static KeyType key_from_value(ValueType const& value) { return value.first; }

  using InsertionResult = std::pair<typename std::map<K, V>::iterator, bool>;
  static auto get_iterator(InsertionResult const& result) { return result.first; }

  template <class Allocator>
  using rebind_alloc = std::map<K, V, std::less<K>, Allocator>;
};

int main(int argc, char** argv) {
  support::associative_container_benchmarks<std::map<int, int>>("std::map<int, int>");
  support::associative_container_benchmarks<std::map<std::string, int>>("std::map<std::string, int>");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
