//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <unordered_set>
#include <string>
#include <random>
#include <vector>

#include "../../GenerateInput.h"
#include "benchmark/benchmark.h"

// Generate random strings of at least 32 chars
struct LongStringGenerator {
  static std::vector<std::string> cached_strings;

  static void ensure_strings(size_t count) {
    cached_strings.clear();

    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<size_t> len_dist(32, 128);

    cached_strings.reserve(count);
    for (size_t i = 0; i < count; i++) {
      std::string str(len_dist(gen), 0);
      for (char& c : str) {
        c = 'a' + (gen() % 26);
      }
      cached_strings.push_back(std::move(str));
    }
  }

  const std::string& generate(size_t i) { return cached_strings[i]; }
};

std::vector<std::string> LongStringGenerator::cached_strings;
[[maybe_unused]] auto dummy = [] { // Pre-generate 32K strings
  LongStringGenerator::ensure_strings(1 << 15);
  return 0;
}();

template <class Gen>
static void BM_UnorderedSet_Find_EmptyNoBuckets(benchmark::State& state, Gen g) {
  const size_t lookup_count = state.range(0);
  std::unordered_set<std::string> s; // Empty and no buckets

  for (auto _ : state) {
    for (size_t i = 0; i < lookup_count; i++) {
      benchmark::DoNotOptimize(s.find(g.generate(i)));
    }
  }
}

template <class Gen>
static void BM_UnorderedSet_Find_EmptyWithBuckets(benchmark::State& state, Gen g) {
  const size_t lookup_count = state.range(0);
  std::unordered_set<std::string> s;
  s.reserve(1); // Still empty but reserved buckets

  for (auto _ : state) {
    for (size_t i = 0; i < lookup_count; i++) {
      benchmark::DoNotOptimize(s.find(g.generate(i)));
    }
  }
}

template <class Gen>
static void BM_UnorderedSet_Find_NonEmpty(benchmark::State& state, Gen g) {
  const size_t lookup_count = state.range(0);
  std::unordered_set<std::string> s{"hello"};

  for (auto _ : state) {
    for (size_t i = 0; i < lookup_count; i++) {
      benchmark::DoNotOptimize(s.find(g.generate(i)));
    }
  }
}

BENCHMARK_CAPTURE(BM_UnorderedSet_Find_EmptyNoBuckets, long_string, LongStringGenerator())
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 15); // Test from 1K to 32K lookups

BENCHMARK_CAPTURE(BM_UnorderedSet_Find_EmptyWithBuckets, long_string, LongStringGenerator())
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 15);

BENCHMARK_CAPTURE(BM_UnorderedSet_Find_NonEmpty, long_string, LongStringGenerator())
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 15);

BENCHMARK_MAIN();
