//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <codecvt>

// template <class Elem, unsigned long Maxcode = 0x10ffff,
//           codecvt_mode Mode = (codecvt_mode)0>
// class codecvt_utf16
//     : public codecvt<Elem, char, mbstate_t>
// {
//     // unspecified
// };

// int length(stateT& state, const externT* from, const externT* from_end,
//            size_t max) const;

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
    typedef std::codecvt_utf16<char16_t> C;
    C c;
    char n[4] = {char(0xD8), char(0xC0), char(0xDC), char(0x03)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 2);
    assert(r == 0);

    n[0] = char(0x10);
    n[1] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x04);
    n[1] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x00);
    n[1] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char16_t, 0x1000> C;
    C c;
    char n[4] = {char(0xD8), char(0xC0), char(0xDC), char(0x03)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 2);
    assert(r == 0);

    n[0] = char(0x10);
    n[1] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 0);

    n[0] = char(0x04);
    n[1] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x00);
    n[1] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char16_t, 0x10ffff, std::consume_header> C;
    C c;
    char n[6] = {char(0xFE), char(0xFF), char(0xD8), char(0xC0), char(0xDC), char(0x03)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 6, 2);
    assert(r == 2);

    n[0] = char(0x10);
    n[1] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x04);
    n[1] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x00);
    n[1] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char16_t, 0x10ffff, std::little_endian> C;
    C c;
    char n[4] = {char(0xC0), char(0xD8), char(0x03), char(0xDC)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 2);
    assert(r == 0);

    n[1] = char(0x10);
    n[0] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x04);
    n[0] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x00);
    n[0] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char16_t, 0x1000, std::little_endian> C;
    C c;
    char n[4] = {char(0xC0), char(0xD8), char(0x03), char(0xDC)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 2);
    assert(r == 0);

    n[1] = char(0x10);
    n[0] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 0);

    n[1] = char(0x04);
    n[0] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x00);
    n[0] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char16_t, 0x10ffff, std::codecvt_mode(std::consume_header | std::little_endian)> C;
    C c;
    char n[6] = {char(0xFF), char(0xFE), char(0xC0), char(0xD8), char(0x03), char(0xDC)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 6, 2);
    assert(r == 2);

    n[1] = char(0x10);
    n[0] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x04);
    n[0] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x00);
    n[0] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
}

template <class CharT>
void TestHelper<CharT, 4>::test() {
  {
    typedef std::codecvt_utf16<char32_t> C;
    C c;
    char n[4] = {char(0xD8), char(0xC0), char(0xDC), char(0x03)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 2);
    assert(r == 4);

    n[0] = char(0x10);
    n[1] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x04);
    n[1] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x00);
    n[1] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char32_t, 0x1000> C;
    C c;
    char n[4] = {char(0xD8), char(0xC0), char(0xDC), char(0x03)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 2);
    assert(r == 0);

    n[0] = char(0x10);
    n[1] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 0);

    n[0] = char(0x04);
    n[1] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x00);
    n[1] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char32_t, 0x10ffff, std::consume_header> C;
    C c;
    char n[6] = {char(0xFE), char(0xFF), char(0xD8), char(0xC0), char(0xDC), char(0x03)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 6, 2);
    assert(r == 6);

    n[0] = char(0x10);
    n[1] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x04);
    n[1] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[0] = char(0x00);
    n[1] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char32_t, 0x10ffff, std::little_endian> C;
    C c;
    char n[4] = {char(0xC0), char(0xD8), char(0x03), char(0xDC)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 2);
    assert(r == 4);

    n[1] = char(0x10);
    n[0] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x04);
    n[0] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x00);
    n[0] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char32_t, 0x1000, std::little_endian> C;
    C c;
    char n[4] = {char(0xC0), char(0xD8), char(0x03), char(0xDC)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 4, 2);
    assert(r == 0);

    n[1] = char(0x10);
    n[0] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 0);

    n[1] = char(0x04);
    n[0] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x00);
    n[0] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
  {
    typedef std::codecvt_utf16<char32_t, 0x10ffff, std::codecvt_mode(std::consume_header | std::little_endian)> C;
    C c;
    char n[6] = {char(0xFF), char(0xFE), char(0xC0), char(0xD8), char(0x03), char(0xDC)};
    std::mbstate_t m;
    int r = c.length(m, n, n + 6, 2);
    assert(r == 6);

    n[1] = char(0x10);
    n[0] = char(0x05);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x04);
    n[0] = char(0x53);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);

    n[1] = char(0x00);
    n[0] = char(0x56);
    r = c.length(m, n, n + 2, 2);
    assert(r == 2);
  }
}

int main(int, char**) {
  TestHelper<wchar_t>::test();
  TestHelper<char16_t>::test();
  TestHelper<char32_t>::test();
  return 0;
}
