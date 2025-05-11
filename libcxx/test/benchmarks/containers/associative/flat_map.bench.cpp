//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <flat_map>
#include <utility>
#include <ranges>

#include "associative_container_benchmarks.h"
#include "../../GenerateInput.h"
#include "benchmark/benchmark.h"

template <class K, class V>
struct support::adapt_operations<std::flat_map<K, V>> {
  using ValueType = typename std::flat_map<K, V>::value_type;
  using KeyType   = typename std::flat_map<K, V>::key_type;
  static ValueType value_from_key(KeyType const& k) { return {k, Generate<V>::arbitrary()}; }
  static KeyType key_from_value(ValueType const& value) { return value.first; }

  using InsertionResult = std::pair<typename std::flat_map<K, V>::iterator, bool>;
  static auto get_iterator(InsertionResult const& result) { return result.first; }
};

void product_iterator_benchmark_flat_map(benchmark::State& state) {
  const std::size_t size = state.range(0);

  using M = std::flat_map<int, int>;

  const M source =
      std::views::iota(0, static_cast<int>(size)) | std::views::transform([](int i) { return std::pair(i, i); }) |
      std::ranges::to<std::flat_map<int, int>>();

  for (auto _ : state) {
    M m;
    m.insert(std::sorted_unique, source.begin(), source.end());
    benchmark::DoNotOptimize(m);
    benchmark::ClobberMemory();
  }
}

void product_iterator_benchmark_zip_view(benchmark::State& state) {
  const std::size_t size = state.range(0);

  using M = std::flat_map<int, int>;

  const std::vector<int> keys   = std::views::iota(0, static_cast<int>(size)) | std::ranges::to<std::vector<int>>();
  const std::vector<int> values = keys;

  auto source = std::views::zip(keys, values);
  for (auto _ : state) {
    M m;
    m.insert(std::sorted_unique, source.begin(), source.end());
    benchmark::DoNotOptimize(m);
    benchmark::ClobberMemory();
  }
}

int main(int argc, char** argv) {
  support::associative_container_benchmarks<std::flat_map<int, int>>("std::flat_map<int, int>");

  benchmark::RegisterBenchmark("flat_map::insert_product_iterator_flat_map", product_iterator_benchmark_flat_map)
      ->Arg(32)
      ->Arg(1024)
      ->Arg(8192)
      ->Arg(65536);
  benchmark::RegisterBenchmark("flat_map::insert_product_iterator_zip", product_iterator_benchmark_zip_view)
      ->Arg(32)
      ->Arg(1024)
      ->Arg(8192)
      ->Arg(65536);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
