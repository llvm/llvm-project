//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <format>

#include <string>

#include "benchmark/benchmark.h"
#include "make_string.h"
#include "test_macros.h"

#define CSTR(S) MAKE_CSTRING(CharT, S)

template <class CharT>
static void BM_format_string(benchmark::State& state) {
  size_t size = state.range(0);
  std::basic_string<CharT> str(size, CharT('*'));

  while (state.KeepRunningBatch(str.size())) {
    std::basic_string<CharT> s = std::format(CSTR("{}"), str);
    benchmark::DoNotOptimize(s);
  }

  state.SetBytesProcessed(state.iterations() * size * sizeof(CharT));
}
BENCHMARK(BM_format_string<char>)->RangeMultiplier(2)->Range(1, 1 << 20);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
BENCHMARK(BM_format_string<wchar_t>)->RangeMultiplier(2)->Range(1, 1 << 20);
#endif

BENCHMARK_MAIN();
