//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include <locale>

#include <benchmark/benchmark.h>

#include "make_string.h"
#include "test_macros.h"

template <class CharT>
static void BM_tolower_char(benchmark::State& state) {
  const auto& ct = std::use_facet<std::ctype<CharT>>(std::locale::classic());

  for (auto _ : state) {
    CharT c('c');
    benchmark::DoNotOptimize(c);
    benchmark::DoNotOptimize(ct.tolower(c));
  }
}

BENCHMARK(BM_tolower_char<char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
BENCHMARK(BM_tolower_char<wchar_t>);
#endif

template <class CharT>
static void BM_tolower_string(benchmark::State& state) {
  const auto& ct = std::use_facet<std::ctype<CharT>>(std::locale::classic());
  std::basic_string<CharT> str;

  for (auto _ : state) {
    str = MAKE_STRING_VIEW(CharT, "THIS IS A LONG STRING TO MAKE TO LOWER");
    benchmark::DoNotOptimize(str);
    benchmark::DoNotOptimize(ct.tolower(str.data(), str.data() + str.size()));
  }
}

BENCHMARK(BM_tolower_string<char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
BENCHMARK(BM_tolower_string<wchar_t>);
#endif

template <class CharT>
static void BM_toupper_char(benchmark::State& state) {
  const auto& ct = std::use_facet<std::ctype<CharT>>(std::locale::classic());

  for (auto _ : state) {
    benchmark::DoNotOptimize(ct.toupper(CharT('c')));
  }
}

BENCHMARK(BM_toupper_char<char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
BENCHMARK(BM_toupper_char<wchar_t>);
#endif

template <class CharT>
static void BM_toupper_string(benchmark::State& state) {
  const auto& ct = std::use_facet<std::ctype<CharT>>(std::locale::classic());
  std::basic_string<CharT> str;

  for (auto _ : state) {
    str = MAKE_STRING_VIEW(CharT, "this is a long string to make to upper");
    benchmark::DoNotOptimize(str);
    benchmark::DoNotOptimize(ct.toupper(str.data(), str.data() + str.size()));
  }
}

BENCHMARK(BM_toupper_string<char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
BENCHMARK(BM_toupper_string<wchar_t>);
#endif

BENCHMARK_MAIN();
