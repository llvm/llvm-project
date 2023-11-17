//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

#include <string>
#include <cassert>

#include "test_macros.h"
#include "asan_testing.h"

template <class S>
void test() {
  S short_s1(3, 'a'), long_s1(100, 'c');
  short_s1.reserve(0x1337);
  long_s1.reserve(0x1337);

  LIBCPP_ASSERT(is_string_asan_correct(short_s1));
  LIBCPP_ASSERT(is_string_asan_correct(long_s1));
#if TEST_STD_VER >= 11
  short_s1.shrink_to_fit();
  long_s1.shrink_to_fit();

  LIBCPP_ASSERT(is_string_asan_correct(short_s1));
  LIBCPP_ASSERT(is_string_asan_correct(long_s1));
#endif
  short_s1.clear();
  long_s1.clear();

  LIBCPP_ASSERT(is_string_asan_correct(short_s1));
  LIBCPP_ASSERT(is_string_asan_correct(long_s1));
#if TEST_STD_VER >= 11
  short_s1.shrink_to_fit();
  long_s1.shrink_to_fit();

  LIBCPP_ASSERT(is_string_asan_correct(short_s1));
  LIBCPP_ASSERT(is_string_asan_correct(long_s1));
#endif
  S short_s2(3, 'a'), long_s2(100, 'c');
  short_s2.reserve(0x1);
  long_s2.reserve(0x1);

  LIBCPP_ASSERT(is_string_asan_correct(short_s2));
  LIBCPP_ASSERT(is_string_asan_correct(long_s2));
}

int main(int, char**) {
  test<std::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::wstring>();
#endif
#if TEST_STD_VER >= 11
  test<std::u16string>();
  test<std::u32string>();
#endif
#if TEST_STD_VER >= 20
  test<std::u8string>();
#endif

  return 0;
}
