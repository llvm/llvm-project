//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

#include <flat_map>

#include "associative_container_benchmarks.h"
#include "../../GenerateInput.h"
#include "benchmark/benchmark.h"

template <class K, class V>
struct support::adapt_operations<std::flat_multimap<K, V>> {
  using ValueType = typename std::flat_multimap<K, V>::value_type;
  using KeyType   = typename std::flat_multimap<K, V>::key_type;
  static ValueType value_from_key(KeyType const& k) { return {k, Generate<V>::arbitrary()}; }
  static KeyType key_from_value(ValueType const& value) { return value.first; }

  using InsertionResult = typename std::flat_multimap<K, V>::iterator;
  static auto get_iterator(InsertionResult const& result) { return result; }
};

int main(int argc, char** argv) {
  support::associative_container_benchmarks<std::flat_multimap<int, int>>("std::flat_multimap<int, int>");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
