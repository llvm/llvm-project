//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// Tests the implementation of the extended grapheme cluster boundaries per
// https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundary_Rules
//
// The tests are based on the test data provided by Unicode
// https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/GraphemeBreakTest.txt

#include <cassert>
#include <format>
#include <functional>
#include <numeric>

#include "test_macros.h"

TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__format/width_estimation_table.h>
TEST_DIAGNOSTIC_POP

// [format.string.std]/12
//
// - U+4DC0 - U+4DFF (Yijing Hexagram Symbols)
// - U+1F300 - U+1F5FF (Miscellaneous Symbols and Pictographs)
// - U+1F900 - U+1F9FF (Supplemental Symbols and Pictographs)
static void constexpr test_hardcoded_values() {
  auto test = [](char32_t c) { assert(std::__width_estimation_table::__estimated_width(c) == 2); };
  for (char32_t c = 0x4DC0; c <= 0x4DFF; ++c)
    test(c);
  for (char32_t c = 0x1F300; c <= 0x1F5FF; ++c)
    test(c);
  for (char32_t c = 0x1F900; c <= 0x1F9FF; ++c)
    test(c);
}

static void constexpr test_invalid_values() {
  auto test = [](char32_t c) { assert(std::__width_estimation_table::__estimated_width(c) == 1); };

  // The surrogate range
  for (char32_t c = 0xD800; c <= 0xDFFF; ++c)
    test(c);

  // The first 256 non valid code points
  for (char32_t c = 0x110000; c <= 0x1100FF; ++c)
    test(c);
}

static void constexpr test_optimization_boundaries() {
  // Entries after the table have a width of 1.
  static_assert(*(std::end(std::__width_estimation_table::__entries) - 1) == ((0x3c000u << 14) | 16381u),
                "validate whether the optimizations in __estimated_width are still valid");
  assert(std::__width_estimation_table::__estimated_width(0x3fffd) == 2);
  assert(std::__width_estimation_table::__estimated_width(0x3fffe) == 1);

  // Entries before the table have a width of 1.
  static_assert(std::__width_estimation_table::__entries[0] >> 14 == 0x1100,
                "validate whether the optimizations in __estimated_width are still valid");
  assert(std::__width_estimation_table::__estimated_width(0x10FF) == 1);
  assert(std::__width_estimation_table::__estimated_width(0x1100) == 2);
}

static constexpr bool test() {
  test_hardcoded_values();
  test_invalid_values();
  test_optimization_boundaries();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
