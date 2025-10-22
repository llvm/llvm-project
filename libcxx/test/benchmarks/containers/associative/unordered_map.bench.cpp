//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <unordered_map>
#include <utility>

#include "associative_container_benchmarks.h"
#include "../../GenerateInput.h"
#include "benchmark/benchmark.h"

static void BM_map_find_string_literal(benchmark::State& state) {
  std::unordered_map<std::string, int> map;
  map.emplace("Something very very long to show a long string situation", 1);
  map.emplace("Something Else", 2);

  for (auto _ : state) {
    benchmark::DoNotOptimize(map);
    benchmark::DoNotOptimize(map.find("Something very very long to show a long string situation"));
  }
}

BENCHMARK(BM_map_find_string_literal);

template <class K, class V>
struct support::adapt_operations<std::unordered_map<K, V>> {
  using ValueType = typename std::unordered_map<K, V>::value_type;
  using KeyType   = typename std::unordered_map<K, V>::key_type;
  static ValueType value_from_key(KeyType const& k) { return {k, Generate<V>::arbitrary()}; }
  static KeyType key_from_value(ValueType const& value) { return value.first; }

  using InsertionResult = std::pair<typename std::unordered_map<K, V>::iterator, bool>;
  static auto get_iterator(InsertionResult const& result) { return result.first; }
};

int main(int argc, char** argv) {
  support::associative_container_benchmarks<std::unordered_map<int, int>>("std::unordered_map<int, int>");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
