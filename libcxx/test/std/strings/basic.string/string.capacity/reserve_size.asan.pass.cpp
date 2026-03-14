//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// This test verifies that the ASan annotations for basic_string objects remain accurate
// after invoking basic_string::reserve(size_type __requested_capacity).
// Different types are used to confirm that ASan works correctly with types of different sizes.
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

  short_s1.clear();
  long_s1.clear();

  LIBCPP_ASSERT(is_string_asan_correct(short_s1));
  LIBCPP_ASSERT(is_string_asan_correct(long_s1));

  short_s1.reserve(0x1);
  long_s1.reserve(0x1);

  LIBCPP_ASSERT(is_string_asan_correct(short_s1));
  LIBCPP_ASSERT(is_string_asan_correct(long_s1));

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
