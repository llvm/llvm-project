//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// This benchmark writes the print and benchmark data to stdout. In order to
// preserve the benchmark output it needs to be stored in a file (using the
// console format). Another issue with the benchmark is the time it takes to
// write output to the real terminal. In order to avoid that overhead write the
// output to a fast "terminal", like /dev/null. For example, the printf
//   console    1546   ns
//   /dev/null    70.9 ns
// An example of a good test invocation.
// BENCHMARK_OUT=benchmark.txt BENCHMARK_OUT_FORMAT=console <exe> >/dev/null

#include <print>

#include <cstdio>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

void printf(benchmark::State& s) {
  while (s.KeepRunning())
    std::printf("The answer to life, the universe, and everything is %d.\n", 42);
}
BENCHMARK(printf);

void vprint_string(std::string_view fmt, std::format_args args) {
  auto s             = std::vformat(fmt, args);
  std::size_t result = fwrite(s.data(), 1, s.size(), stdout);
  if (result < s.size())
    throw std::format_error("fwrite error");
}

template <typename... T>
void print_string(std::format_string<T...> fmt, T&&... args) {
  vprint_string(fmt.get(), std::make_format_args(args...));
}

void print_string(benchmark::State& s) {
  while (s.KeepRunning()) {
    print_string("The answer to life, the universe, and everything is {}.\n", 42);
  }
}
BENCHMARK(print_string);

void vprint_stack(std::string_view fmt, std::format_args args) {
  auto buf = std::vector<char>{};
  std::vformat_to(std::back_inserter(buf), fmt, args);
  std::size_t result = fwrite(buf.data(), 1, buf.size(), stdout);
  if (result < buf.size())
    throw std::format_error("fwrite error");
}

template <typename... T>
void print_stack(std::format_string<T...> fmt, T&&... args) {
  vprint_stack(fmt.get(), std::make_format_args(args...));
}

void print_stack(benchmark::State& s) {
  while (s.KeepRunning()) {
    print_stack("The answer to life, the universe, and everything is {}.\n", 42);
  }
}
BENCHMARK(print_stack);

void print_direct(benchmark::State& s) {
  while (s.KeepRunning())
    std::print("The answer to life, the universe, and everything is {}.\n", 42);
}
BENCHMARK(print_direct);

BENCHMARK_MAIN();
