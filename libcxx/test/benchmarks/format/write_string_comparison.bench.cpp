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
#include <concepts>
#include <cstdio>
#include <deque>
#include <format>
#include <iterator>
#include <list>
#include <string>
#include <string_view>
#include <vector>

#include "benchmark/benchmark.h"
#include "test_macros.h"

const char* c_string_6_characters  = "abcdef";
const char* c_string_60_characters = "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef";
const char* c_string_6000_characters =
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
    "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"; // 100 lines

std::string string_6_characters    = c_string_6_characters;
std::string string_60_characters   = c_string_60_characters;
std::string string_6000_characters = c_string_6000_characters;

std::string_view string_view_6_characters    = c_string_6_characters;
std::string_view string_view_60_characters   = c_string_60_characters;
std::string_view string_view_6000_characters = c_string_6000_characters;

static void BM_sprintf(benchmark::State& state, const char* value) {
  std::array<char, 10'000> output;
  for (auto _ : state)
    benchmark::DoNotOptimize(std::sprintf(output.data(), "%s", value));
}

template <class T>
static void BM_format(benchmark::State& state, const T& value) {
  for (auto _ : state)
    benchmark::DoNotOptimize(std::format("{}", value));
}

template <class Container, class T>
static void BM_format_to_back_inserter(benchmark::State& state, const T& value) {
  for (auto _ : state) {
    Container c;
    std::format_to(std::back_inserter(c), "{}", value);
    benchmark::DoNotOptimize(c);
  }
}

template <class T, class F>
static void BM_format_to_iterator(benchmark::State& state, const T& value, F&& f) {
  auto output = f();
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::format_to(std::begin(output), "{}", value));
  }
}

#define FORMAT_BENCHMARKS(name, variable)                                                                              \
  BENCHMARK_CAPTURE(BM_format, name, variable);                                                                        \
                                                                                                                       \
  BENCHMARK_TEMPLATE1_CAPTURE(BM_format_to_back_inserter, std::string, name, variable);                                \
  BENCHMARK_TEMPLATE1_CAPTURE(BM_format_to_back_inserter, std::vector<char>, name, variable);                          \
  BENCHMARK_TEMPLATE1_CAPTURE(BM_format_to_back_inserter, std::deque<char>, name, variable);                           \
  BENCHMARK_TEMPLATE1_CAPTURE(BM_format_to_back_inserter, std::list<char>, name, variable);                            \
                                                                                                                       \
  BENCHMARK_CAPTURE(BM_format_to_iterator, <std::array> name, variable, ([] {                                          \
                      std::array<char, 10'000> a;                                                                      \
                      return a;                                                                                        \
                    }));                                                                                               \
  BENCHMARK_CAPTURE(BM_format_to_iterator, <std::string> name, variable, ([] {                                         \
                      std::string s;                                                                                   \
                      s.resize(10'000);                                                                                \
                      return s;                                                                                        \
                    }));                                                                                               \
  BENCHMARK_CAPTURE(BM_format_to_iterator, <std::vector> name, variable, ([] {                                         \
                      std::vector<char> v;                                                                             \
                      v.resize(10'000);                                                                                \
                      return v;                                                                                        \
                    }));                                                                                               \
  BENCHMARK_CAPTURE(BM_format_to_iterator, <std::deque> name, variable, ([] {                                          \
                      std::deque<char> d;                                                                              \
                      d.resize(10'000);                                                                                \
                      return d;                                                                                        \
                    }));                                                                                               \
                                                                                                                       \
  /* */

BENCHMARK_CAPTURE(BM_sprintf, C_string_len_6, c_string_6_characters);
FORMAT_BENCHMARKS(C_string_len_6, c_string_6_characters)
FORMAT_BENCHMARKS(string_len_6, string_6_characters)
FORMAT_BENCHMARKS(string_view_len_6, string_view_6_characters)

BENCHMARK_CAPTURE(BM_sprintf, C_string_len_60, c_string_60_characters);
FORMAT_BENCHMARKS(C_string_len_60, c_string_60_characters)
FORMAT_BENCHMARKS(string_len_60, string_60_characters)
FORMAT_BENCHMARKS(string_view_len_60, string_view_60_characters)

BENCHMARK_CAPTURE(BM_sprintf, C_string_len_6000, c_string_6000_characters);
FORMAT_BENCHMARKS(C_string_len_6000, c_string_6000_characters)
FORMAT_BENCHMARKS(string_len_6000, string_6000_characters)
FORMAT_BENCHMARKS(string_view_len_6000, string_view_6000_characters)

BENCHMARK_MAIN();
