//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <iterator>

#include "test_iterators.h"
#include <vector>

static std::vector<char> comparable_data;

template <class ElementT>
class TriviallyComparable {
  ElementT el_;

public:
  TEST_CONSTEXPR TriviallyComparable(ElementT el) : el_(el) {}
  bool operator==(const TriviallyComparable&) const = default;
};

template <class IndexT>
class Comparable {
  IndexT index_;

public:
  Comparable(IndexT i)
      : index_([&]() {
          IndexT size = static_cast<IndexT>(comparable_data.size());
          comparable_data.push_back(i);
          return size;
        }()) {}

  bool operator==(const Comparable& other) const {
    return comparable_data[other.index_] == comparable_data[index_];
  }

  friend bool operator==(const Comparable& lhs, long long rhs) { return comparable_data[lhs.index_] == rhs; }
};

static void bm_contains(benchmark::State& state) {
  std::vector<char> a(state.range(), 'a');

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);

    benchmark::DoNotOptimize(std::ranges::contains(a.begin(), a.end(), 'a'));
  }
}
BENCHMARK(bm_contains)->RangeMultiplier(16)->Range(16, 16 << 20);

static void bm_contains_with_trivially_comparable(benchmark::State& state) {
  std::vector<TriviallyComparable<char>> a(state.range(), 'a');

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);

    benchmark::DoNotOptimize(std::ranges::contains(a.begin(), a.end(), 'a'));
  }
}
BENCHMARK(bm_contains_with_trivially_comparable)->RangeMultiplier(16)->Range(16, 16 << 20);

static void bm_contains_with_comparable(benchmark::State& state) {
  std::vector<Comparable<char>> a(state.range(), 'a');

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);

    benchmark::DoNotOptimize(std::ranges::contains(a.begin(), a.end(), 'a'));
  }
}
BENCHMARK(bm_contains_with_comparable)->RangeMultiplier(16)->Range(16, 16 << 20);

BENCHMARK_MAIN();