//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// byte_string to_bytes(Elem wchar);
// byte_string to_bytes(const Elem* wptr);
// byte_string to_bytes(const wide_string& wstr);
// byte_string to_bytes(const Elem* first, const Elem* last);

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
    std::wstring_convert<std::codecvt_utf8<CharT> > myconv;
    std::wstring ws(1, L'\u1005');
    std::string bs = myconv.to_bytes(ws[0]);
    assert(bs == "\xE1\x80\x85");
    bs = myconv.to_bytes(ws.c_str());
    assert(bs == "\xE1\x80\x85");
    bs = myconv.to_bytes(ws);
    assert(bs == "\xE1\x80\x85");
    bs = myconv.to_bytes(ws.data(), ws.data() + ws.size());
    assert(bs == "\xE1\x80\x85");
    bs = myconv.to_bytes(L"");
    assert(bs.size() == 0);
  }
}

template <class CharT>
void TestHelper<CharT, 4>::test() {
  static_assert((std::is_same<CharT, wchar_t>::value), "");
  {
    std::wstring_convert<std::codecvt_utf8<CharT> > myconv;
    std::wstring ws(1, *L"\U00040003");
    std::string bs = myconv.to_bytes(ws[0]);
    assert(bs == "\xF1\x80\x80\x83");
    bs = myconv.to_bytes(ws.c_str());
    assert(bs == "\xF1\x80\x80\x83");
    bs = myconv.to_bytes(ws);
    assert(bs == "\xF1\x80\x80\x83");
    bs = myconv.to_bytes(ws.data(), ws.data() + ws.size());
    assert(bs == "\xF1\x80\x80\x83");
    bs = myconv.to_bytes(L"");
    assert(bs.size() == 0);
  }
}

int main(int, char**) {
  TestHelper<wchar_t>::test();
  return 0;
}
