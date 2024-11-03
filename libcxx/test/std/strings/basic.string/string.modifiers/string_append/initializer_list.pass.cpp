//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string& append(initializer_list<charT> il); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "nasty_string.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test() {
  using CharT = typename S::value_type;

  S s(CONVERT_TO_CSTRING(CharT, "123"));
  s.append({CharT('a'), CharT('b'), CharT('c')});
  assert(s == CONVERT_TO_CSTRING(CharT, "123abc"));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test<std::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::wstring>();
#endif
#if TEST_STD_VER >= 20
  test<std::u8string>();
#endif
  test<std::u16string>();
  test<std::u32string>();

  test<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_string>();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
