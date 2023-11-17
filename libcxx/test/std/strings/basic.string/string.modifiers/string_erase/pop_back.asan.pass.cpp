//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

#include <string>
#include <cassert>

#include "test_macros.h"
#include "asan_testing.h"

template <class CharT>
void test(const CharT val) {
  using S = std::basic_string<CharT>;

  S s;
  while (s.size() < 8000) {
    s.push_back(val);

    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
  while (s.size() > 0) {
    s.pop_back();

    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
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
