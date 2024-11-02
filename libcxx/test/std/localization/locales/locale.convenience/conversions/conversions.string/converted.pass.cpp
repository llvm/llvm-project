//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// size_t converted() const;

// XFAIL: no-wide-characters

#include <locale>
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
  static_assert((std::is_same<CharT, wchar_t>::value), "");
  {
    typedef std::codecvt_utf8<CharT> Codecvt;
    typedef std::wstring_convert<Codecvt> Myconv;
    Myconv myconv;
    assert(myconv.converted() == 0);
    std::string bs = myconv.to_bytes(L"\u1005");
    assert(myconv.converted() == 1);
    bs = myconv.to_bytes(L"\u1005e");
    assert(myconv.converted() == 2);
    std::wstring ws = myconv.from_bytes("\xE1\x80\x85");
    assert(myconv.converted() == 3);
  }
}

template <class CharT>
void TestHelper<CharT, 4>::test() {
  static_assert((std::is_same<CharT, wchar_t>::value), "");
  {
    typedef std::codecvt_utf8<CharT> Codecvt;
    typedef std::wstring_convert<Codecvt> Myconv;
    Myconv myconv;
    assert(myconv.converted() == 0);
    std::string bs = myconv.to_bytes(L"\U00040003");
    assert(myconv.converted() == 1);
    bs = myconv.to_bytes(L"\U00040003e");
    assert(myconv.converted() == 2);
    std::wstring ws = myconv.from_bytes("\xF1\x80\x80\x83");
    assert(myconv.converted() == 4);
  }
}

int main(int, char**) {
  TestHelper<wchar_t>::test();
  return 0;
}
