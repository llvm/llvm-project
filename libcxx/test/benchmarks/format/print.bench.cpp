//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <print>

#include "benchmark/benchmark.h"

#include "test_macros.h"

template <class CharT>
static void BM_string_without_formatting(benchmark::State& state) {
  std::FILE* devnull = std::fopen("/dev/null", "w");
  for (auto _ : state) {
    std::print(devnull, "Hello, World!");
  }
}
BENCHMARK(BM_string_without_formatting<char>)->Name("std::print(\"Hello, World!\")");

#ifndef TEST_HAS_NO_UNICODE
template <class CharT>
static void BM_string_without_formatting_opaque(benchmark::State& state) {
  std::FILE* devnull = std::fopen("/dev/null", "w");
  for (auto _ : state) {
    const char* format_string = "Hello, World!";
    benchmark::DoNotOptimize(format_string);
    std::vprint_unicode(devnull, format_string, std::make_format_args());
  }
}
BENCHMARK(BM_string_without_formatting_opaque<char>)->Name("std::vprint_unicode(\"Hello, World!\")");
#endif

BENCHMARK_MAIN();
