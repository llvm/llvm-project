//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=10000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=70000000

// template<container-compatible-range<charT> R>
//   constexpr basic_string& replace_with_range(const_iterator i1, const_iterator i2, R&& rg); // C++23

#include <string>
#include <utility>

#include "../../../../containers/sequences/insert_range_sequence_containers.h"
#include "test_macros.h"

template <class Range>
concept HasReplaceWithRange = requires (std::string& c, Range&& range) {
  c.replace_with_range(c.end(), c.end(), std::forward<Range>(range));
};

constexpr bool test_constraints_replace_with_range() {
  // Input range with the same value type.
  static_assert(HasReplaceWithRange<InputRange<char>>);
  // Input range with a convertible value type.
  static_assert(HasReplaceWithRange<InputRange<unsigned char>>);
  // Input range with a non-convertible value type.
  static_assert(!HasReplaceWithRange<InputRange<Empty>>);
  // Not an input range.
  static_assert(!HasReplaceWithRange<InputRangeNotDerivedFrom>);
  static_assert(!HasReplaceWithRange<InputRangeNotIndirectlyReadable>);
  static_assert(!HasReplaceWithRange<InputRangeNotInputOrOutputIterator>);

  return true;
}

using StrBuffer = Buffer<char, 256>;

struct TestCaseReplacement {
  StrBuffer initial;
  std::size_t from = 0;
  std::size_t to = 0;
  StrBuffer input;
  StrBuffer expected;
};

// Permutation matrix:
// - initial string: empty / one-element / n elements;
// - cut starts from: beginning / middle / end;
// - cut size: empty / one-element / several elements / until the end;
// - input range: empty / one-element / middle-sized / longer than SSO / longer than the current string capacity.

// Empty string.

constexpr TestCaseReplacement EmptyString_End_EmptyCut_EmptyRange {
  .initial = "", .from = 0, .to = 0, .input = "", .expected = ""
};

constexpr TestCaseReplacement EmptyString_End_EmptyCut_OneElementRange {
  .initial = "", .from = 0, .to = 0, .input = "a", .expected = "a"
};

constexpr TestCaseReplacement EmptyString_End_EmptyCut_MidRange {
  .initial = "", .from = 0, .to = 0, .input = "aeiou", .expected = "aeiou"
};

constexpr TestCaseReplacement EmptyString_End_EmptyCut_LongRange {
  .initial = "", .from = 0, .to = 0,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789"
};

// One-element string.

constexpr TestCaseReplacement OneElementString_Begin_EmptyCut_EmptyRange {
  .initial = "B", .from = 0, .to = 0, .input = "", .expected = "B"
};

constexpr TestCaseReplacement OneElementString_Begin_EmptyCut_OneElementRange {
  .initial = "B", .from = 0, .to = 0, .input = "a", .expected = "aB"
};

constexpr TestCaseReplacement OneElementString_Begin_EmptyCut_MidRange {
  .initial = "B", .from = 0, .to = 0, .input = "aeiou", .expected = "aeiouB"
};

constexpr TestCaseReplacement OneElementString_Begin_EmptyCut_LongRange {
  .initial = "B", .from = 0, .to = 0,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789B"
};

constexpr TestCaseReplacement OneElementString_Begin_OneElementCut_EmptyRange {
  .initial = "B", .from = 0, .to = 1, .input = "", .expected = ""
};

constexpr TestCaseReplacement OneElementString_Begin_OneElementCut_OneElementRange {
  .initial = "B", .from = 0, .to = 1, .input = "a", .expected = "a"
};

constexpr TestCaseReplacement OneElementString_Begin_OneElementCut_MidRange {
  .initial = "B", .from = 0, .to = 1, .input = "aeiou", .expected = "aeiou"
};

constexpr TestCaseReplacement OneElementString_Begin_OneElementCut_LongRange {
  .initial = "B", .from = 0, .to = 1,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789"
};

constexpr TestCaseReplacement OneElementString_End_EmptyCut_EmptyRange {
  .initial = "B", .from = 1, .to = 1, .input = "", .expected = "B"
};

constexpr TestCaseReplacement OneElementString_End_EmptyCut_OneElementRange {
  .initial = "B", .from = 1, .to = 1, .input = "a", .expected = "Ba"
};

constexpr TestCaseReplacement OneElementString_End_EmptyCut_MidRange {
  .initial = "B", .from = 1, .to = 1, .input = "aeiou", .expected = "Baeiou"
};

constexpr TestCaseReplacement OneElementString_End_EmptyCut_LongRange {
  .initial = "B", .from = 1, .to = 1,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "Babcdefghijklmnopqrstuvwxyz0123456789"
};

// Short string (using SSO).

// Replace at the beginning.

constexpr TestCaseReplacement ShortString_Begin_EmptyCut_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 0, .input = "", .expected = "_BCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_EmptyCut_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 0, .input = "a", .expected = "a_BCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_EmptyCut_MidRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 0, .input = "aeiou", .expected = "aeiou_BCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_EmptyCut_LongRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 0,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789_BCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_OneElementCut_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 1, .input = "", .expected = "BCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_OneElementCut_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 1, .input = "a", .expected = "aBCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_OneElementCut_MidRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 1, .input = "aeiou", .expected = "aeiouBCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_OneElementCut_LongRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 1,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789BCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_MidSizedCut_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 3, .input = "", .expected = "DFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_MidSizedCut_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 3, .input = "a", .expected = "aDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_MidSizedCut_MidRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 3, .input = "aeiou", .expected = "aeiouDFGHJ_"
};

// Note: this test case switches from SSO to non-SSO.
constexpr TestCaseReplacement ShortString_Begin_MidSizedCut_LongRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 3,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789DFGHJ_"
};

constexpr TestCaseReplacement ShortString_Begin_CutToEnd_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 9, .input = "", .expected = ""
};

constexpr TestCaseReplacement ShortString_Begin_CutToEnd_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 9, .input = "a", .expected = "a"
};

constexpr TestCaseReplacement ShortString_Begin_CutToEnd_MidRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 9, .input = "aeiou", .expected = "aeiou"
};

// Note: this test case switches from SSO to non-SSO.
constexpr TestCaseReplacement ShortString_Begin_CutToEnd_LongRange {
  .initial = "_BCDFGHJ_", .from = 0, .to = 9,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789"
};

// Replace in the middle.

constexpr TestCaseReplacement ShortString_Mid_EmptyCut_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 4, .input = "", .expected = "_BCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_Mid_EmptyCut_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 4, .input = "a", .expected = "_BCDaFGHJ_"
};

constexpr TestCaseReplacement ShortString_Mid_EmptyCut_MidRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 4, .input = "aeiou", .expected = "_BCDaeiouFGHJ_"
};

constexpr TestCaseReplacement ShortString_Mid_EmptyCut_LongRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 4,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_BCDabcdefghijklmnopqrstuvwxyz0123456789FGHJ_"
};

constexpr TestCaseReplacement ShortString_Mid_OneElementCut_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 5, .input = "", .expected = "_BCDGHJ_"
};

constexpr TestCaseReplacement ShortString_Mid_OneElementCut_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 5, .input = "a", .expected = "_BCDaGHJ_"
};

constexpr TestCaseReplacement ShortString_Mid_OneElementCut_MidRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 5, .input = "aeiou", .expected = "_BCDaeiouGHJ_"
};

constexpr TestCaseReplacement ShortString_Mid_OneElementCut_LongRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 5,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_BCDabcdefghijklmnopqrstuvwxyz0123456789GHJ_"
};

constexpr TestCaseReplacement ShortString_Mid_MidSizedCut_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 7, .input = "", .expected = "_BCDJ_"
};

constexpr TestCaseReplacement ShortString_Mid_MidSizedCut_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 7, .input = "a", .expected = "_BCDaJ_"
};

constexpr TestCaseReplacement ShortString_Mid_MidSizedCut_MidRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 7, .input = "aeiou", .expected = "_BCDaeiouJ_"
};

constexpr TestCaseReplacement ShortString_Mid_MidSizedCut_LongRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 7,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_BCDabcdefghijklmnopqrstuvwxyz0123456789J_"
};

constexpr TestCaseReplacement ShortString_Mid_CutToEnd_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 9, .input = "", .expected = "_BCD"
};

constexpr TestCaseReplacement ShortString_Mid_CutToEnd_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 9, .input = "a", .expected = "_BCDa"
};

constexpr TestCaseReplacement ShortString_Mid_CutToEnd_MidRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 9, .input = "aeiou", .expected = "_BCDaeiou"
};

constexpr TestCaseReplacement ShortString_Mid_CutToEnd_LongRange {
  .initial = "_BCDFGHJ_", .from = 4, .to = 9,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_BCDabcdefghijklmnopqrstuvwxyz0123456789"
};

// Replace at the end.

constexpr TestCaseReplacement ShortString_End_EmptyCut_EmptyRange {
  .initial = "_BCDFGHJ_", .from = 9, .to = 9, .input = "", .expected = "_BCDFGHJ_"
};

constexpr TestCaseReplacement ShortString_End_EmptyCut_OneElementRange {
  .initial = "_BCDFGHJ_", .from = 9, .to = 9, .input = "a", .expected = "_BCDFGHJ_a"
};

constexpr TestCaseReplacement ShortString_End_EmptyCut_MidRange {
  .initial = "_BCDFGHJ_", .from = 9, .to = 9, .input = "aeiou", .expected = "_BCDFGHJ_aeiou"
};

constexpr TestCaseReplacement ShortString_End_EmptyCut_LongRange {
  .initial = "_BCDFGHJ_", .from = 9, .to = 9,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_BCDFGHJ_abcdefghijklmnopqrstuvwxyz0123456789"
};

// Long string (no SSO).

// Replace at the beginning.

constexpr TestCaseReplacement LongString_Begin_EmptyCut_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 0, .input = "",
  .expected = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_EmptyCut_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 0, .input = "a",
  .expected = "a_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_EmptyCut_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 0, .input = "aeiou",
  .expected = "aeiou_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_EmptyCut_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 0,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_EmptyCut_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 0,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
              "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_OneElementCut_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 1, .input = "",
  .expected = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_OneElementCut_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 1, .input = "a",
  .expected = "aABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_OneElementCut_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 1, .input = "aeiou",
  .expected = "aeiouABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_OneElementCut_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 1,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_OneElementCut_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 1,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
              "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_MidSizedCut_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 3, .input = "",
  .expected = "CDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_MidSizedCut_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 3, .input = "a",
  .expected = "aCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_MidSizedCut_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 3, .input = "aeiou",
  .expected = "aeiouCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_MidSizedCut_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 3,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789CDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_MidSizedCut_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 3,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
              "CDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Begin_CutToEnd_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 38, .input = "",
  .expected = ""
};

constexpr TestCaseReplacement LongString_Begin_CutToEnd_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 38, .input = "a",
  .expected = "a"
};

constexpr TestCaseReplacement LongString_Begin_CutToEnd_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 38, .input = "aeiou",
  .expected = "aeiou"
};

constexpr TestCaseReplacement LongString_Begin_CutToEnd_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 38,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789"
};

constexpr TestCaseReplacement LongString_Begin_CutToEnd_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 0, .to = 38,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
};

// Replace in the middle.

constexpr TestCaseReplacement LongString_Mid_EmptyCut_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 18, .input = "",
  .expected = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_EmptyCut_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 18, .input = "a",
  .expected = "_ABCDEFGHIJKLMNOPQaRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_EmptyCut_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 18, .input = "aeiou",
  .expected = "_ABCDEFGHIJKLMNOPQaeiouRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_EmptyCut_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 18,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQabcdefghijklmnopqrstuvwxyz0123456789RSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_EmptyCut_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 18,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQ"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
              "RSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_OneElementCut_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 19, .input = "",
  .expected = "_ABCDEFGHIJKLMNOPQSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_OneElementCut_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 19, .input = "a",
  .expected = "_ABCDEFGHIJKLMNOPQaSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_OneElementCut_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 19, .input = "aeiou",
  .expected = "_ABCDEFGHIJKLMNOPQaeiouSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_OneElementCut_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 19,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQabcdefghijklmnopqrstuvwxyz0123456789STUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_OneElementCut_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 19,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQ"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
              "STUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_MidSizedCut_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 21, .input = "",
  .expected = "_ABCDEFGHIJKLMNOPQUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_MidSizedCut_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 21, .input = "a",
  .expected = "_ABCDEFGHIJKLMNOPQaUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_MidSizedCut_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 21, .input = "aeiou",
  .expected = "_ABCDEFGHIJKLMNOPQaeiouUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_MidSizedCut_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 21,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQabcdefghijklmnopqrstuvwxyz0123456789UVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_MidSizedCut_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 21,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQ"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
              "UVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_Mid_CutToEnd_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 38, .input = "",
  .expected = "_ABCDEFGHIJKLMNOPQ"
};

constexpr TestCaseReplacement LongString_Mid_CutToEnd_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 38, .input = "a",
  .expected = "_ABCDEFGHIJKLMNOPQa"
};

constexpr TestCaseReplacement LongString_Mid_CutToEnd_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 38, .input = "aeiou",
  .expected = "_ABCDEFGHIJKLMNOPQaeiou"
};

constexpr TestCaseReplacement LongString_Mid_CutToEnd_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 38,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQabcdefghijklmnopqrstuvwxyz0123456789"
};

constexpr TestCaseReplacement LongString_Mid_CutToEnd_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 18, .to = 38,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQ"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
};

// Replace at the end.

constexpr TestCaseReplacement LongString_End_EmptyCut_EmptyRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 38, .to = 38, .input = "",
  .expected = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
};

constexpr TestCaseReplacement LongString_End_EmptyCut_OneElementRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 38, .to = 38, .input = "a",
  .expected = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_a"
};

constexpr TestCaseReplacement LongString_End_EmptyCut_MidRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 38, .to = 38, .input = "aeiou",
  .expected = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_aeiou"
};

constexpr TestCaseReplacement LongString_End_EmptyCut_LongRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 38, .to = 38,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_abcdefghijklmnopqrstuvwxyz0123456789"
};

constexpr TestCaseReplacement LongString_End_EmptyCut_LongerThanCapacityRange {
  .initial = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", .from = 38, .to = 38,
  .input = "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789_"
           "abcdefghijklmnopqrstuvwxyz0123456789",
  .expected = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789_"
              "abcdefghijklmnopqrstuvwxyz0123456789"
};

template <class Iter, class Sent, class Alloc>
constexpr void test_string_replace_with_range() {
  auto test = [&](const TestCaseReplacement& test_case) {
    using Container = std::basic_string<char, std::char_traits<char>, Alloc>;

    auto get_pos = [](auto& c, auto index) { return std::ranges::next(c.begin(), index); };
    Container c(test_case.initial.begin(), test_case.initial.end());
    auto in = wrap_input<Iter, Sent>(test_case.input);
    auto from = get_pos(c, test_case.from);
    auto to = get_pos(c, test_case.to);

    Container& result = c.replace_with_range(from, to, in);
    assert(&result == &c);
    LIBCPP_ASSERT(c.__invariants());
    return std::ranges::equal(c, test_case.expected);
  };

  { // Empty string.
    // empty_str.replace_with_range(end, end, empty_range)
    assert(test(EmptyString_End_EmptyCut_EmptyRange));
    // empty_str.replace_with_range(end, end, one_element_range)
    assert(test(EmptyString_End_EmptyCut_OneElementRange));
    // empty_str.replace_with_range(end, end, mid_range)
    assert(test(EmptyString_End_EmptyCut_MidRange));
    // empty_str.replace_with_range(end, end, long_range)
    assert(test(EmptyString_End_EmptyCut_LongRange));
  }

  { // One-element string.
    // one_element_str.replace_with_range(begin, begin, empty_range)
    assert(test(OneElementString_Begin_EmptyCut_EmptyRange));
    // one_element_str.replace_with_range(begin, begin, one_element_range)
    assert(test(OneElementString_Begin_EmptyCut_OneElementRange));
    // one_element_str.replace_with_range(begin, begin, mid_range)
    assert(test(OneElementString_Begin_EmptyCut_MidRange));
    // one_element_str.replace_with_range(begin, begin, long_range)
    assert(test(OneElementString_Begin_EmptyCut_LongRange));
    // one_element_str.replace_with_range(begin, begin + 1, empty_range)
    assert(test(OneElementString_Begin_OneElementCut_EmptyRange));
    // one_element_str.replace_with_range(begin, begin + 1, one_element_range)
    assert(test(OneElementString_Begin_OneElementCut_OneElementRange));
    // one_element_str.replace_with_range(begin, begin + 1, mid_range)
    assert(test(OneElementString_Begin_OneElementCut_MidRange));
    // one_element_str.replace_with_range(begin, begin + 1, long_range)
    assert(test(OneElementString_Begin_OneElementCut_LongRange));
    // one_element_str.replace_with_range(end, end, empty_range)
    assert(test(OneElementString_End_EmptyCut_EmptyRange));
    // one_element_str.replace_with_range(end, end, one_element_range)
    assert(test(OneElementString_End_EmptyCut_OneElementRange));
    // one_element_str.replace_with_range(end, end, mid_range)
    assert(test(OneElementString_End_EmptyCut_MidRange));
    // one_element_str.replace_with_range(end, end, long_range)
    assert(test(OneElementString_End_EmptyCut_LongRange));
  }

  { // Short string.
    // Replace at the beginning.

    // short_str.replace_with_range(begin, begin, empty_range)
    assert(test(ShortString_Begin_EmptyCut_EmptyRange));
    // short_str.replace_with_range(begin, begin, one_element_range)
    assert(test(ShortString_Begin_EmptyCut_OneElementRange));
    // short_str.replace_with_range(begin, begin, mid_range)
    assert(test(ShortString_Begin_EmptyCut_MidRange));
    // short_str.replace_with_range(begin, begin, long_range)
    assert(test(ShortString_Begin_EmptyCut_LongRange));
    // short_str.replace_with_range(begin, begin + 1, empty_range)
    assert(test(ShortString_Begin_OneElementCut_EmptyRange));
    // short_str.replace_with_range(begin, begin + 1, one_element_range)
    assert(test(ShortString_Begin_OneElementCut_OneElementRange));
    // short_str.replace_with_range(begin, begin + 1, mid_range)
    assert(test(ShortString_Begin_OneElementCut_MidRange));
    // short_str.replace_with_range(begin, begin + 1, long_range)
    assert(test(ShortString_Begin_OneElementCut_LongRange));
    // short_str.replace_with_range(begin, begin + 3, empty_range)
    assert(test(ShortString_Begin_MidSizedCut_EmptyRange));
    // short_str.replace_with_range(begin, begin + 3, one_element_range)
    assert(test(ShortString_Begin_MidSizedCut_OneElementRange));
    // short_str.replace_with_range(begin, begin + 3, mid_range)
    assert(test(ShortString_Begin_MidSizedCut_MidRange));
    // short_str.replace_with_range(begin, begin + 3, long_range)
    assert(test(ShortString_Begin_MidSizedCut_LongRange));
    // short_str.replace_with_range(begin, end, empty_range)
    assert(test(ShortString_Begin_CutToEnd_EmptyRange));
    // short_str.replace_with_range(begin, end, one_element_range)
    assert(test(ShortString_Begin_CutToEnd_OneElementRange));
    // short_str.replace_with_range(begin, end, mid_range)
    assert(test(ShortString_Begin_CutToEnd_MidRange));
    // short_str.replace_with_range(begin, end, long_range)
    assert(test(ShortString_Begin_CutToEnd_LongRange));

    // Replace in the middle.

    // short_str.replace_with_range(mid, mid, empty_range)
    assert(test(ShortString_Mid_EmptyCut_EmptyRange));
    // short_str.replace_with_range(mid, mid, one_element_range)
    assert(test(ShortString_Mid_EmptyCut_OneElementRange));
    // short_str.replace_with_range(mid, mid, mid_range)
    assert(test(ShortString_Mid_EmptyCut_MidRange));
    // short_str.replace_with_range(mid, mid, long_range)
    assert(test(ShortString_Mid_EmptyCut_LongRange));
    // short_str.replace_with_range(mid, mid + 1, empty_range)
    assert(test(ShortString_Mid_OneElementCut_EmptyRange));
    // short_str.replace_with_range(mid, mid + 1, one_element_range)
    assert(test(ShortString_Mid_OneElementCut_OneElementRange));
    // short_str.replace_with_range(mid, mid + 1, mid_range)
    assert(test(ShortString_Mid_OneElementCut_MidRange));
    // short_str.replace_with_range(mid, mid + 1, long_range)
    assert(test(ShortString_Mid_OneElementCut_LongRange));
    // short_str.replace_with_range(mid, mid + 3, empty_range)
    assert(test(ShortString_Mid_MidSizedCut_EmptyRange));
    // short_str.replace_with_range(mid, mid + 3, one_element_range)
    assert(test(ShortString_Mid_MidSizedCut_OneElementRange));
    // short_str.replace_with_range(mid, mid + 3, mid_range)
    assert(test(ShortString_Mid_MidSizedCut_MidRange));
    // short_str.replace_with_range(mid, mid + 3, long_range)
    assert(test(ShortString_Mid_MidSizedCut_LongRange));
    // short_str.replace_with_range(mid, end, empty_range)
    assert(test(ShortString_Mid_CutToEnd_EmptyRange));
    // short_str.replace_with_range(mid, end, one_element_range)
    assert(test(ShortString_Mid_CutToEnd_OneElementRange));
    // short_str.replace_with_range(mid, end, mid_range)
    assert(test(ShortString_Mid_CutToEnd_MidRange));
    // short_str.replace_with_range(mid, end, long_range)
    assert(test(ShortString_Mid_CutToEnd_LongRange));

    // Replace at the end.

    // short_str.replace_with_range(end, end, empty_range)
    assert(test(ShortString_End_EmptyCut_EmptyRange));
    // short_str.replace_with_range(end, end, one_element_range)
    assert(test(ShortString_End_EmptyCut_OneElementRange));
    // short_str.replace_with_range(end, end, mid_range)
    assert(test(ShortString_End_EmptyCut_MidRange));
    // short_str.replace_with_range(end, end, long_range)
    assert(test(ShortString_End_EmptyCut_LongRange));
  }

  { // Long string.
    // Replace at the beginning.

    // long_str.replace_with_range(begin, begin, empty_range)
    assert(test(LongString_Begin_EmptyCut_EmptyRange));
    // long_str.replace_with_range(begin, begin, one_element_range)
    assert(test(LongString_Begin_EmptyCut_OneElementRange));
    // long_str.replace_with_range(begin, begin, mid_range)
    assert(test(LongString_Begin_EmptyCut_MidRange));
    // long_str.replace_with_range(begin, begin, long_range)
    assert(test(LongString_Begin_EmptyCut_LongRange));
    // long_str.replace_with_range(begin, begin, longer_than_capacity_range)
    assert(test(LongString_Begin_EmptyCut_LongerThanCapacityRange));
    // long_str.replace_with_range(begin, begin + 1, empty_range)
    assert(test(LongString_Begin_OneElementCut_EmptyRange));
    // long_str.replace_with_range(begin, begin + 1, one_element_range)
    assert(test(LongString_Begin_OneElementCut_OneElementRange));
    // long_str.replace_with_range(begin, begin + 1, mid_range)
    assert(test(LongString_Begin_OneElementCut_MidRange));
    // long_str.replace_with_range(begin, begin + 1, long_range)
    assert(test(LongString_Begin_OneElementCut_LongRange));
    // long_str.replace_with_range(begin, begin + 1, longer_than_capacity_range)
    assert(test(LongString_Begin_OneElementCut_LongerThanCapacityRange));
    // long_str.replace_with_range(begin, begin + 3, empty_range)
    assert(test(LongString_Begin_MidSizedCut_EmptyRange));
    // long_str.replace_with_range(begin, begin + 3, one_element_range)
    assert(test(LongString_Begin_MidSizedCut_OneElementRange));
    // long_str.replace_with_range(begin, begin + 3, mid_range)
    assert(test(LongString_Begin_MidSizedCut_MidRange));
    // long_str.replace_with_range(begin, begin + 3, long_range)
    assert(test(LongString_Begin_MidSizedCut_LongRange));
    // long_str.replace_with_range(begin, begin + 3, longer_than_capacity_range)
    assert(test(LongString_Begin_MidSizedCut_LongerThanCapacityRange));
    // long_str.replace_with_range(begin, end, empty_range)
    assert(test(LongString_Begin_CutToEnd_EmptyRange));
    // long_str.replace_with_range(begin, end, one_element_range)
    assert(test(LongString_Begin_CutToEnd_OneElementRange));
    // long_str.replace_with_range(begin, end, mid_range)
    assert(test(LongString_Begin_CutToEnd_MidRange));
    // long_str.replace_with_range(begin, end, long_range)
    assert(test(LongString_Begin_CutToEnd_LongRange));
    // long_str.replace_with_range(begin, end, longer_than_capacity_range)
    assert(test(LongString_Begin_CutToEnd_LongerThanCapacityRange));

    // Replace in the middle.

    // long_str.replace_with_range(mid, mid, empty_range)
    assert(test(LongString_Mid_EmptyCut_EmptyRange));
    // long_str.replace_with_range(mid, mid, one_element_range)
    assert(test(LongString_Mid_EmptyCut_OneElementRange));
    // long_str.replace_with_range(mid, mid, mid_range)
    assert(test(LongString_Mid_EmptyCut_MidRange));
    // long_str.replace_with_range(mid, mid, long_range)
    assert(test(LongString_Mid_EmptyCut_LongRange));
    // long_str.replace_with_range(mid, mid, longer_than_capacity_range)
    assert(test(LongString_Mid_EmptyCut_LongerThanCapacityRange));
    // long_str.replace_with_range(mid, mid + 1, empty_range)
    assert(test(LongString_Mid_OneElementCut_EmptyRange));
    // long_str.replace_with_range(mid, mid + 1, one_element_range)
    assert(test(LongString_Mid_OneElementCut_OneElementRange));
    // long_str.replace_with_range(mid, mid + 1, mid_range)
    assert(test(LongString_Mid_OneElementCut_MidRange));
    // long_str.replace_with_range(mid, mid + 1, long_range)
    assert(test(LongString_Mid_OneElementCut_LongRange));
    // long_str.replace_with_range(mid, mid + 1, longer_than_capacity_range)
    assert(test(LongString_Mid_OneElementCut_LongerThanCapacityRange));
    // long_str.replace_with_range(mid, mid + 3, empty_range)
    assert(test(LongString_Mid_MidSizedCut_EmptyRange));
    // long_str.replace_with_range(mid, mid + 3, one_element_range)
    assert(test(LongString_Mid_MidSizedCut_OneElementRange));
    // long_str.replace_with_range(mid, mid + 3, mid_range)
    assert(test(LongString_Mid_MidSizedCut_MidRange));
    // long_str.replace_with_range(mid, mid + 3, long_range)
    assert(test(LongString_Mid_MidSizedCut_LongRange));
    // long_str.replace_with_range(mid, mid + 3, longer_than_capacity_range)
    assert(test(LongString_Mid_MidSizedCut_LongerThanCapacityRange));
    // long_str.replace_with_range(mid, end, empty_range)
    assert(test(LongString_Mid_CutToEnd_EmptyRange));
    // long_str.replace_with_range(mid, end, one_element_range)
    assert(test(LongString_Mid_CutToEnd_OneElementRange));
    // long_str.replace_with_range(mid, end, mid_range)
    assert(test(LongString_Mid_CutToEnd_MidRange));
    // long_str.replace_with_range(mid, end, long_range)
    assert(test(LongString_Mid_CutToEnd_LongRange));
    // long_str.replace_with_range(mid, end, longer_than_capacity_range)
    assert(test(LongString_Mid_CutToEnd_LongerThanCapacityRange));

    // Replace at the end.

    // long_str.replace_with_range(end, end, empty_range)
    assert(test(LongString_End_EmptyCut_EmptyRange));
    // long_str.replace_with_range(end, end, one_element_range)
    assert(test(LongString_End_EmptyCut_OneElementRange));
    // long_str.replace_with_range(end, end, mid_range)
    assert(test(LongString_End_EmptyCut_MidRange));
    // long_str.replace_with_range(end, end, long_range)
    assert(test(LongString_End_EmptyCut_LongRange));
    // long_str.replace_with_range(end, end, longer_than_capacity_range)
    assert(test(LongString_End_EmptyCut_LongerThanCapacityRange));
  }
}

constexpr void test_string_replace_with_range_rvalue_range() {
  // Make sure that the input range can be passed by both an lvalue and an rvalue reference and regardless of constness.

  { // Lvalue.
    std::string in = "abc";
    std::string c = "123";
    c.replace_with_range(c.begin(), c.end(), in);
  }

  { // Const lvalue.
    const std::string in = "abc";
    std::string c = "123";
    c.replace_with_range(c.begin(), c.end(), in);
  }

  { // Rvalue.
    std::string in = "abc";
    std::string c = "123";
    c.replace_with_range(c.begin(), c.end(), std::move(in));
  }

  { // Const rvalue.
    const std::string in = "abc";
    std::string c = "123";
    c.replace_with_range(c.begin(), c.end(), std::move(in));
  }
}

constexpr bool test_constexpr() {
  for_all_iterators_and_allocators_constexpr<char, const char*>([]<class Iter, class Sent, class Alloc>() {
    test_string_replace_with_range<Iter, Sent, Alloc>();
  });
  test_string_replace_with_range_rvalue_range();

  return true;
}

// Tested cases:
// - different kinds of replacements (varying the size of the initial string, the cut point and the cut size, and the
//   size of the input range);
// - an exception is thrown when allocating new elements.
int main(int, char**) {
  static_assert(test_constraints_replace_with_range());

  for_all_iterators_and_allocators<char, const char*>([]<class Iter, class Sent, class Alloc>() {
    test_string_replace_with_range<Iter, Sent, Alloc>();
  });
  test_string_replace_with_range_rvalue_range();
  static_assert(test_constexpr());

  // Note: `test_string_replace_with_range_exception_safety_throwing_copy` doesn't apply because copying chars cannot
  // throw.
  {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    // Note: the input string must be long enough to prevent SSO, otherwise the allocator won't be used.
    std::string in(64, 'a');

    try {
      ThrowingAllocator<char> alloc;

      globalMemCounter.reset();
      std::basic_string<char, std::char_traits<char>, ThrowingAllocator<char>> c(alloc);
      c.replace_with_range(c.end(), c.end(), in);
      assert(false); // The function call above should throw.

    } catch (int) {
      assert(globalMemCounter.new_called == globalMemCounter.delete_called);
    }
#endif
  }

  return 0;
}
