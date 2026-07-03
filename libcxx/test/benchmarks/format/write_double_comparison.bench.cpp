//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Don't warn about std::sprintf
// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated

#include <array>
#include <charconv>
#include <cstdio>
#include <format>
#include <iterator>
#include <list>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"

std::array data = [] {
  std::uniform_real_distribution<double> distribution;
  std::mt19937 generator;
  std::array<double, 1000> result;
  std::generate_n(result.begin(), result.size(), [&] { return distribution(generator); });
  return result;
}();

static void BM_sprintf(benchmark::State& state) {
  std::array<char, 100> output;
  while (state.KeepRunningBatch(data.size()))
    for (auto value : data) {
      std::sprintf(output.data(), "%f", value);
      benchmark::DoNotOptimize(output.data());
    }
}

static void BM_to_string(benchmark::State& state) {
  while (state.KeepRunningBatch(data.size()))
    for (auto value : data) {
      std::string s = std::to_string(value);
      benchmark::DoNotOptimize(s);
    }
}

static void BM_to_chars(benchmark::State& state) {
  std::array<char, 100> output;

  while (state.KeepRunningBatch(data.size()))
    for (auto value : data) {
      std::to_chars(output.data(), output.data() + output.size(), value);
      benchmark::DoNotOptimize(output.data());
    }
}

static void BM_to_chars_as_string(benchmark::State& state) {
  std::array<char, 100> output;

  while (state.KeepRunningBatch(data.size()))
    for (auto value : data) {
      char* end = std::to_chars(output.data(), output.data() + output.size(), value).ptr;
      std::string s{output.data(), end};
      benchmark::DoNotOptimize(s);
    }
}

static void BM_format(benchmark::State& state) {
  while (state.KeepRunningBatch(data.size()))
    for (auto value : data) {
      std::string s = std::format("{}", value);
      benchmark::DoNotOptimize(s);
    }
}

template <class C>
static void BM_format_to_back_inserter(benchmark::State& state) {
  while (state.KeepRunningBatch(data.size()))
    for (auto value : data) {
      C c;
      std::format_to(std::back_inserter(c), "{}", value);
      benchmark::DoNotOptimize(c);
    }
}

template <class F>
static void BM_format_to_iterator(benchmark::State& state, F&& f) {
  auto output = f();
  while (state.KeepRunningBatch(data.size()))
    for (auto value : data) {
      std::format_to(std::begin(output), "{}", value);
      benchmark::DoNotOptimize(std::begin(output));
    }
}

BENCHMARK(BM_sprintf);
BENCHMARK(BM_to_string);
BENCHMARK(BM_to_chars);
BENCHMARK(BM_to_chars_as_string);
BENCHMARK(BM_format);
BENCHMARK_TEMPLATE(BM_format_to_back_inserter, std::string);
BENCHMARK_TEMPLATE(BM_format_to_back_inserter, std::vector<char>);
BENCHMARK_TEMPLATE(BM_format_to_back_inserter, std::list<char>);
BENCHMARK_CAPTURE(BM_format_to_iterator, <std::array>, ([] {
                    std::array<char, 100> a;
                    return a;
                  }));
BENCHMARK_CAPTURE(BM_format_to_iterator, <std::string>, ([] {
                    std::string s;
                    s.resize(100);
                    return s;
                  }));
BENCHMARK_CAPTURE(BM_format_to_iterator, <std::vector>, ([] {
                    std::vector<char> v;
                    v.resize(100);
                    return v;
                  }));

BENCHMARK_MAIN();
