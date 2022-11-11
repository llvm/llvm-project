//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <codecvt>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// template <class Elem, unsigned long Maxcode = 0x10ffff,
//           codecvt_mode Mode = (codecvt_mode)0>
// class codecvt_utf16
//     : public codecvt<Elem, char, mbstate_t>
// {
//     // unspecified
// };

// int max_length() const throw();

#include <codecvt>
#include <cassert>

#include "test_macros.h"

template <class CharT, size_t = sizeof(CharT)>
struct TestHelper;

template <class CharT>
struct TestHelper<CharT, 2> {
  static void test();
};

template <class CharT>
struct TestHelper<CharT, 4> {
  static void test();
};

template <class CharT>
void TestHelper<CharT, 2>::test() {
  {
    typedef std::codecvt_utf16<CharT> C;
    C c;
    int r = c.max_length();
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<CharT, 0xFFFFFFFF, std::consume_header> C;
    C c;
    int r = c.max_length();
    assert(r == 4);
  }
}

template <class CharT>
void TestHelper<CharT, 4>::test() {
  {
    typedef std::codecvt_utf16<CharT> C;
    C c;
    int r = c.max_length();
    assert(r == 4);
  }
  {
    typedef std::codecvt_utf16<CharT, 0xFFFFFFFF, std::consume_header> C;
    C c;
    int r = c.max_length();
    assert(r == 6);
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  TestHelper<wchar_t>::test();
#endif
  TestHelper<char16_t>::test();
  TestHelper<char32_t>::test();
  return 0;
}
