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
// class codecvt_utf8
//     : public codecvt<Elem, char, mbstate_t>
// {
//     // unspecified
// };

// int length(stateT& state, const externT* from, const externT* from_end,
//            size_t max) const;

#include <codecvt>
#include <cassert>

#include "test_macros.h"

template <class CharT, std::size_t = sizeof(CharT)>
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
    typedef std::codecvt_utf8<CharT> C;
    C c;
    char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 1);
    assert(r == 0);

    n[0] = char(0xE1);
    n[1] = char(0x80);
    n[2] = char(0x85);
    r = c.length(m, n, n + 3, 2);
    assert(r == 3);

    n[0] = char(0xD1);
    n[1] = char(0x93);
    r = c.length(m, n, n + 2, 3);
    assert(r == 2);

    n[0] = char(0x56);
    r = c.length(m, n, n + 1, 3);
    assert(r == 1);
  }
  {
    typedef std::codecvt_utf8<CharT, 0x1000> C;
    C c;
    char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 1);
    assert(r == 0);

    n[0] = char(0xE1);
    n[1] = char(0x80);
    n[2] = char(0x85);
    r = c.length(m, n, n + 3, 2);
    assert(r == 0);

    n[0] = char(0xD1);
    n[1] = char(0x93);
    r = c.length(m, n, n + 2, 3);
    assert(r == 2);

    n[0] = char(0x56);
    r = c.length(m, n, n + 1, 3);
    assert(r == 1);
  }
  {
    typedef std::codecvt_utf8<CharT, 0xFFFFFFFF, std::consume_header> C;
    C c;
    char n[7] = {char(0xEF), char(0xBB), char(0xBF), char(0xF1), char(0x80), char(0x80), char(0x83)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 7, 1);
    assert(r == 3);

    n[0] = char(0xE1);
    n[1] = char(0x80);
    n[2] = char(0x85);
    r = c.length(m, n, n + 3, 2);
    assert(r == 3);

    n[0] = char(0xEF);
    n[1] = char(0xBB);
    n[2] = char(0xBF);
    n[3] = char(0xD1);
    n[4] = char(0x93);
    r = c.length(m, n, n + 5, 3);
    assert(r == 5);

    n[0] = char(0x56);
    r = c.length(m, n, n + 1, 3);
    assert(r == 1);
  }
}

template <class CharT>
void TestHelper<CharT, 4>::test() {
  {
    typedef std::codecvt_utf8<CharT> C;
    C c;
    char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 1);
    assert(r == 4);

    n[0] = char(0xE1);
    n[1] = char(0x80);
    n[2] = char(0x85);
    r = c.length(m, n, n + 3, 2);
    assert(r == 3);

    n[0] = char(0xD1);
    n[1] = char(0x93);
    r = c.length(m, n, n + 2, 3);
    assert(r == 2);

    n[0] = char(0x56);
    r = c.length(m, n, n + 1, 3);
    assert(r == 1);
  }
  {
    typedef std::codecvt_utf8<CharT, 0x1000> C;
    C c;
    char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 1);
    assert(r == 0);

    n[0] = char(0xE1);
    n[1] = char(0x80);
    n[2] = char(0x85);
    r = c.length(m, n, n + 3, 2);
    assert(r == 0);

    n[0] = char(0xD1);
    n[1] = char(0x93);
    r = c.length(m, n, n + 2, 3);
    assert(r == 2);

    n[0] = char(0x56);
    r = c.length(m, n, n + 1, 3);
    assert(r == 1);
  }
  {
    typedef std::codecvt_utf8<CharT, 0xFFFFFFFF, std::consume_header> C;
    C c;
    char n[7] = {char(0xEF), char(0xBB), char(0xBF), char(0xF1), char(0x80), char(0x80), char(0x83)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 7, 1);
    assert(r == 7);

    n[0] = char(0xE1);
    n[1] = char(0x80);
    n[2] = char(0x85);
    r = c.length(m, n, n + 3, 2);
    assert(r == 3);

    n[0] = char(0xEF);
    n[1] = char(0xBB);
    n[2] = char(0xBF);
    n[3] = char(0xD1);
    n[4] = char(0x93);
    r = c.length(m, n, n + 5, 3);
    assert(r == 5);

    n[0] = char(0x56);
    r = c.length(m, n, n + 1, 3);
    assert(r == 1);
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  TestHelper<wchar_t>::test();
#endif
  TestHelper<char32_t>::test();
  TestHelper<char16_t>::test();
  return 0;
}
