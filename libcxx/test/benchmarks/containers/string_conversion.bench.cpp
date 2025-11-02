//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <limits>
#include <string>

#include <benchmark/benchmark.h>

template <class CharT>
void string_to_arithmetic_impl(int& out, const std::basic_string<CharT>& str) {
  out = std::stoi(str);
}

template <class CharT>
void string_to_arithmetic_impl(long& out, const std::basic_string<CharT>& str) {
  out = std::stol(str);
}

template <class CharT>
void string_to_arithmetic_impl(long long& out, const std::basic_string<CharT>& str) {
  out = std::stoll(str);
}

template <class CharT>
void string_to_arithmetic_impl(unsigned long& out, const std::basic_string<CharT>& str) {
  out = std::stoul(str);
}

template <class CharT>
void string_to_arithmetic_impl(unsigned long long& out, const std::basic_string<CharT>& str) {
  out = std::stoull(str);
}

template <class CharT>
void string_to_arithmetic_impl(float& out, const std::basic_string<CharT>& str) {
  out = std::stof(str);
}

template <class CharT>
void string_to_arithmetic_impl(double& out, const std::basic_string<CharT>& str) {
  out = std::stod(str);
}

template <class CharT>
void string_to_arithmetic_impl(long double& out, const std::basic_string<CharT>& str) {
  out = std::stold(str);
}

template <class Integer>
std::string to_string_dispatch(char, Integer i) {
  return std::to_string(i);
}

template <class Integer>
std::wstring to_string_dispatch(wchar_t, Integer i) {
  return std::to_wstring(i);
}

template <class CharT, class Integer>
void BM_string_to_arithmetic(benchmark::State& state) {
  std::basic_string<CharT> num = to_string_dispatch(CharT(), std::numeric_limits<Integer>::max());

  for (auto _ : state) {
    benchmark::DoNotOptimize(num);
    Integer val;
    string_to_arithmetic_impl(val, num);
  }
}

BENCHMARK(BM_string_to_arithmetic<char, int>);
BENCHMARK(BM_string_to_arithmetic<char, long>);
BENCHMARK(BM_string_to_arithmetic<char, long long>);
BENCHMARK(BM_string_to_arithmetic<char, unsigned long>);
BENCHMARK(BM_string_to_arithmetic<char, unsigned long long>);
BENCHMARK(BM_string_to_arithmetic<char, float>);
BENCHMARK(BM_string_to_arithmetic<char, double>);
BENCHMARK(BM_string_to_arithmetic<char, long double>);
BENCHMARK(BM_string_to_arithmetic<wchar_t, int>);
BENCHMARK(BM_string_to_arithmetic<wchar_t, long>);
BENCHMARK(BM_string_to_arithmetic<wchar_t, long long>);
BENCHMARK(BM_string_to_arithmetic<wchar_t, unsigned long>);
BENCHMARK(BM_string_to_arithmetic<wchar_t, unsigned long long>);
BENCHMARK(BM_string_to_arithmetic<char, float>);
BENCHMARK(BM_string_to_arithmetic<char, double>);
BENCHMARK(BM_string_to_arithmetic<char, long double>);

BENCHMARK_MAIN();
