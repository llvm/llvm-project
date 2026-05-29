//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cstddef>
#include <string>

#include "benchmark/benchmark.h"

constexpr std::size_t MAX_STRING_LEN = 8 << 14;

// No match found — worst case for find_first_of (must scan entire haystack).
// Varies haystack size with a fixed needle size.
static void BM_StringFindFirstOfNoMatch(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle("xyz!@#$%");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfNoMatch)->Range(32, MAX_STRING_LEN);

// No match found — varies needle size with a fixed haystack.
// Demonstrates O(n*m) vs O(n+m) scaling with needle size.
static void BM_StringFindFirstOfNoMatchVaryNeedle(benchmark::State& state) {
  std::string haystack(8192, 'a');
  std::string needle(state.range(0), 'b');
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfNoMatchVaryNeedle)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);

// Match at the end — must scan nearly the entire haystack before finding it.
static void BM_StringFindFirstOfMatchAtEnd(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  haystack.back() = 'z';
  std::string needle("xyz!@#$%");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfMatchAtEnd)->Range(32, MAX_STRING_LEN);

// find_last_of — no match, must scan entire haystack backward.
static void BM_StringFindLastOfNoMatch(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle("xyz!@#$%");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_last_of(needle));
  }
}
BENCHMARK(BM_StringFindLastOfNoMatch)->Range(32, MAX_STRING_LEN);

// find_first_not_of — all characters are in the needle, no "not of" found.
static void BM_StringFindFirstNotOfNoMatch(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle("abcdefgh");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_not_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstNotOfNoMatch)->Range(32, MAX_STRING_LEN);

// find_last_not_of — all characters are in the needle, no "not of" found.
static void BM_StringFindLastNotOfNoMatch(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle("abcdefgh");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_last_not_of(needle));
  }
}
BENCHMARK(BM_StringFindLastNotOfNoMatch)->Range(32, MAX_STRING_LEN);

BENCHMARK_MAIN();
