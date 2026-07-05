//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// This test validates that ASan annotations are correctly updated after swaps.
// This test meticulously interchanges objects that store data both within their internal memory (Short
// String Optimization) and in external buffers (non-SSO).

#include <string>
#include <cassert>

#include "test_macros.h"
#include "asan_testing.h"

template <class CharT>
void test(const CharT val) {
  using S = std::basic_string<CharT>;

  S empty_s;
  S short_s(3, val);
  S long_s(1100, val);

  std::swap(empty_s, empty_s);
  std::swap(short_s, short_s);
  std::swap(long_s, long_s);
  LIBCPP_ASSERT(is_string_asan_correct(empty_s));
  LIBCPP_ASSERT(is_string_asan_correct(short_s));
  LIBCPP_ASSERT(is_string_asan_correct(long_s));

  std::swap(empty_s, short_s);
  LIBCPP_ASSERT(is_string_asan_correct(empty_s));
  LIBCPP_ASSERT(is_string_asan_correct(short_s));

  std::swap(empty_s, short_s);
  LIBCPP_ASSERT(is_string_asan_correct(empty_s));
  LIBCPP_ASSERT(is_string_asan_correct(short_s));

  std::swap(empty_s, long_s);
  LIBCPP_ASSERT(is_string_asan_correct(empty_s));
  LIBCPP_ASSERT(is_string_asan_correct(long_s));

  std::swap(empty_s, long_s);
  LIBCPP_ASSERT(is_string_asan_correct(empty_s));
  LIBCPP_ASSERT(is_string_asan_correct(long_s));

  std::swap(short_s, long_s);
  LIBCPP_ASSERT(is_string_asan_correct(short_s));
  LIBCPP_ASSERT(is_string_asan_correct(long_s));

  std::swap(short_s, long_s);
  LIBCPP_ASSERT(is_string_asan_correct(short_s));
  LIBCPP_ASSERT(is_string_asan_correct(long_s));

  S long_s2(11100, val);

  std::swap(long_s, long_s2);
  LIBCPP_ASSERT(is_string_asan_correct(long_s));
  LIBCPP_ASSERT(is_string_asan_correct(long_s2));

  std::swap(long_s, long_s2);
  LIBCPP_ASSERT(is_string_asan_correct(long_s));
  LIBCPP_ASSERT(is_string_asan_correct(long_s2));

  S long_s3(111, val);

  std::swap(long_s, long_s3);
  LIBCPP_ASSERT(is_string_asan_correct(long_s));
  LIBCPP_ASSERT(is_string_asan_correct(long_s2));
}

int main(int, char**) {
  test<char>('x');
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>(L'x');
#endif
#if TEST_STD_VER >= 11
  test<char16_t>(u'x');
  test<char32_t>(U'x');
#endif
#if TEST_STD_VER >= 20
  test<char8_t>(u8'x');
#endif

  return 0;
}
