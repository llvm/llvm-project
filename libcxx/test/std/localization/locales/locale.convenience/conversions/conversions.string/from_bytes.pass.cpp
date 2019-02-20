//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wide_string from_bytes(char byte);
// wide_string from_bytes(const char* ptr);
// wide_string from_bytes(const byte_string& str);
// wide_string from_bytes(const char* first, const char* last);

#include <locale>
#include <codecvt>
#include <cassert>

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
    std::string bs("\xE1\x80\x85\x00");
    std::wstring ws = myconv.from_bytes('a');
    assert(ws == L"a");
    ws = myconv.from_bytes(bs.c_str());
    assert(ws == L"\x1005");
    ws = myconv.from_bytes(bs);
    assert(ws == L"\x1005");
    ws = myconv.from_bytes(bs.data(), bs.data() + bs.size());
    assert(ws == L"\x1005");
    ws = myconv.from_bytes("");
    assert(ws.size() == 0);
  }
}

template <class CharT>
void TestHelper<CharT, 4>::test() {
  static_assert((std::is_same<CharT, wchar_t>::value), "");
  {
    std::wstring_convert<std::codecvt_utf8<CharT> > myconv;
    std::string bs("\xF1\x80\x80\x83");
    std::wstring ws = myconv.from_bytes('a');
    assert(ws == L"a");
    ws = myconv.from_bytes(bs.c_str());
    assert(ws == L"\x40003");
    ws = myconv.from_bytes(bs);
    assert(ws == L"\x40003");
    ws = myconv.from_bytes(bs.data(), bs.data() + bs.size());
    assert(ws == L"\x40003");
    ws = myconv.from_bytes("");
    assert(ws.size() == 0);
  }
}

int main(int, char**) { TestHelper<wchar_t>::test(); 
  return 0;
}
