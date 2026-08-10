//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf

// basic_string_view<charT, traits> view() const noexcept;

#include <sstream>
#include <cassert>
#include <type_traits>

#include "make_string.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
struct my_char_traits : public std::char_traits<CharT> {};

template <class CharT>
static void test() {
  std::basic_stringbuf<CharT> buf(STR("testing"));
  static_assert(noexcept(buf.view()));
  assert(buf.view() == SV("testing"));
  buf.str(STR("another test"));
  assert(buf.view() == SV("another test"));

  std::basic_stringbuf<CharT> robuf(STR("foo"), std::ios_base::in);
  assert(robuf.view() == SV("foo"));

  std::basic_stringbuf<CharT> nbuf(STR("not used"), 0);
  assert(nbuf.view() == std::basic_string_view<CharT>());

  const std::basic_stringbuf<CharT> cbuf(STR("abc"));
  static_assert(noexcept(cbuf.view()));
  assert(cbuf.view() == SV("abc"));

  std::basic_stringbuf<CharT, my_char_traits<CharT>> tbuf;
  static_assert(std::is_same_v<decltype(tbuf.view()), std::basic_string_view<CharT, my_char_traits<CharT>>>);
}

struct StringBuf : std::stringbuf {
  using basic_stringbuf::basic_stringbuf;
  void public_setg(int a, int b, int c) {
    char* p = eback();
    this->setg(p + a, p + b, p + c);
  }
};

static void test_altered_sequence_pointers() {
  {
    auto src = StringBuf("hello world", std::ios_base::in);
    src.public_setg(4, 6, 9);
    std::stringbuf dest;
    dest = std::move(src);
    assert(dest.view() == dest.str());
    LIBCPP_ASSERT(dest.view() == "o wor");
  }
  {
    auto src = StringBuf("hello world", std::ios_base::in);
    src.public_setg(4, 6, 9);
    std::stringbuf dest;
    dest.swap(src);
    assert(dest.view() == dest.str());
    LIBCPP_ASSERT(dest.view() == "o wor");
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  test_altered_sequence_pointers();
  return 0;
}
