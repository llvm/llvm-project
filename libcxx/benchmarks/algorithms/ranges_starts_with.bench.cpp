//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <vector>

#include "test_iterators.h"

static void bm_starts_with_contiguous_iter_with_memcmp_optimization(benchmark::State& state) {
  std::vector<int> a(state.range(), 1);
  std::vector<int> p(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(p);

    auto begin1 = contiguous_iterator(a.data());
    auto end1   = contiguous_iterator(a.data() + a.size());
    auto begin2 = contiguous_iterator(p.data());
    auto end2   = contiguous_iterator(p.data() + p.size());

    benchmark::DoNotOptimize(std::ranges::starts_with(begin1, end1, begin2, end2));
  }
}
BENCHMARK(bm_starts_with_contiguous_iter_with_memcmp_optimization)->RangeMultiplier(16)->Range(16, 16 << 20);

static void bm_starts_with_contiguous_iter(benchmark::State& state) {
  std::vector<int> a(state.range(), 1);
  std::vector<int> p(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(p);

    auto begin1 = contiguous_iterator(a.data());
    auto end1   = contiguous_iterator(a.data() + a.size());
    auto begin2 = contiguous_iterator(p.data());
    auto end2   = contiguous_iterator(p.data() + p.size());

    // Using a custom predicate to make sure the memcmp optimization doesn't get invoked
    benchmark::DoNotOptimize(
        std::ranges::starts_with(begin1, end1, begin2, end2, [](const int a, const int b) { return a == b; }));
  }
}
BENCHMARK(bm_starts_with_contiguous_iter)->RangeMultiplier(16)->Range(16, 16 << 20);

static void bm_starts_with_random_access_iter(benchmark::State& state) {
  std::vector<int> a(state.range(), 1);
  std::vector<int> p(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(p);

    auto begin1 = random_access_iterator(a.data());
    auto end1   = random_access_iterator(a.data() + a.size());
    auto begin2 = random_access_iterator(p.data());
    auto end2   = random_access_iterator(p.data() + p.size());

    benchmark::DoNotOptimize(std::ranges::starts_with(begin1, end1, begin2, end2));
  }
}
BENCHMARK(bm_starts_with_random_access_iter)->RangeMultiplier(16)->Range(16, 16 << 20);

static void bm_starts_with_bidirectional_iter(benchmark::State& state) {
  std::vector<int> a(state.range(), 1);
  std::vector<int> p(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(p);

    auto begin1 = bidirectional_iterator(a.data());
    auto end1   = bidirectional_iterator(a.data() + a.size());
    auto begin2 = bidirectional_iterator(p.data());
    auto end2   = bidirectional_iterator(p.data() + p.size());

    benchmark::DoNotOptimize(std::ranges::starts_with(begin1, end1, begin2, end2));
  }
}
BENCHMARK(bm_starts_with_bidirectional_iter)->RangeMultiplier(16)->Range(16, 16 << 20);

static void bm_starts_with_forward_iter(benchmark::State& state) {
  std::vector<int> a(state.range(), 1);
  std::vector<int> p(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(p);

    auto begin1 = forward_iterator(a.data());
    auto end1   = forward_iterator(a.data() + a.size());
    auto begin2 = forward_iterator(p.data());
    auto end2   = forward_iterator(p.data() + p.size());

    benchmark::DoNotOptimize(std::ranges::starts_with(begin1, end1, begin2, end2));
  }
}
BENCHMARK(bm_starts_with_forward_iter)->RangeMultiplier(16)->Range(16, 16 << 20);

BENCHMARK_MAIN();
