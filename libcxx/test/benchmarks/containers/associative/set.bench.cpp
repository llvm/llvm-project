//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <set>
#include <utility>

#include "associative_container_benchmarks.h"
#include "benchmark/benchmark.h"

template <class K>
struct support::adapt_operations<std::set<K>> {
  using ValueType = typename std::set<K>::value_type;
  using KeyType   = typename std::set<K>::key_type;
  static ValueType value_from_key(KeyType const& k) { return k; }
  static KeyType key_from_value(ValueType const& value) { return value; }

  using InsertionResult = std::pair<typename std::set<K>::iterator, bool>;
  static auto get_iterator(InsertionResult const& result) { return result.first; }

  template <class Allocator>
  using rebind_alloc = std::set<K, std::less<K>, Allocator>;
};

int main(int argc, char** argv) {
  support::associative_container_benchmarks<std::set<int>>("std::set<int>");
  support::associative_container_benchmarks<std::set<std::string>>("std::set<std::string>");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
