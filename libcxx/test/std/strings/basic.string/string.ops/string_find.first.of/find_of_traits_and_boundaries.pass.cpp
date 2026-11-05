//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=50000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=200000000

// <string>

// size_type find_first_of(const charT* s, size_type pos, size_type n) const; // constexpr since C++20
// size_type find_last_of(const charT* s, size_type pos, size_type n) const; // constexpr since C++20
// size_type find_first_not_of(const charT* s, size_type pos, size_type n) const; // constexpr since C++20
// size_type find_last_not_of(const charT* s, size_type pos, size_type n) const; // constexpr since C++20
// (and the basic_string overloads)

// Tests the find_first_of family with custom traits (including traits whose
// eq() is not plain equality), high-byte characters, wide characters with
// values of 256 and above, embedded '\0', duplicate needle characters,
// position boundaries, empty and long needles, and matches on both sides of
// index 16.

#include <string>
#include <cassert>
#include <string>

#include "constexpr_char_traits.h"
#include "string_find_of_helpers.h"
#include "test_macros.h"

template <class CharT, class Traits = std::char_traits<CharT> >
using S = std::basic_string<CharT, Traits>;

void test_runtime() {
  test_find_of_family<S<char> >(true);
  test_find_of_family<S<char, ci_char_traits> >(true);
  test_find_of_family<S<char, constexpr_char_traits<char> > >(true);
#ifndef TEST_HAS_NO_CHAR8_T
  test_find_of_family<S<char8_t> >(true);
#endif
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_find_of_family<S<wchar_t> >(true);
#endif
  test_find_of_family<S<char16_t> >(true);
  test_find_of_family<S<char32_t> >(true);

  test_find_of_case_insensitive<S<char, ci_char_traits> >();
}

#if TEST_STD_VER >= 20
// The results during constant evaluation must agree with the runtime results.
constexpr bool test_constexpr() {
  test_find_of_family<S<char> >(false);
  test_find_of_family<S<char, ci_char_traits> >(false);
#  ifndef TEST_HAS_NO_CHAR8_T
  test_find_of_family<S<char8_t> >(false);
#  endif

  test_find_of_case_insensitive<S<char, ci_char_traits> >();
  return true;
}
#endif

int main(int, char**) {
  test_runtime();
#if TEST_STD_VER >= 20
  test_constexpr();
  static_assert(test_constexpr());
#endif

  return 0;
}
