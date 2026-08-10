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

// basic_string<charT, traits, Allocator> str() &&;

#include <sstream>
#include <cassert>

#include "make_string.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class CharT>
static void test() {
  {
    std::basic_stringbuf<CharT> buf(STR("testing"));
    std::basic_string<CharT> s = std::move(buf).str();
    assert(s == STR("testing"));
    assert(buf.view().empty());
  }
  {
    std::basic_stringbuf<CharT> buf;
    std::basic_string<CharT> s = std::move(buf).str();
    assert(s.empty());
    assert(buf.view().empty());
  }
  {
    std::basic_stringbuf<CharT> buf(STR("a very long string that exceeds the small string optimization buffer length"));
    const CharT* p             = buf.view().data();
    std::basic_string<CharT> s = std::move(buf).str();
    assert(s.data() == p); // the allocation was pilfered
    assert(buf.view().empty());
  }
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
    dest             = std::move(src);
    std::string view = std::string(dest.view());
    std::string str  = std::move(dest).str();
    assert(view == str);
    LIBCPP_ASSERT(str == "o wor");
    assert(dest.str().empty());
    assert(dest.view().empty());
  }
  {
    auto src = StringBuf("hello world", std::ios_base::in);
    src.public_setg(4, 6, 9);
    std::stringbuf dest;
    dest.swap(src);
    std::string view = std::string(dest.view());
    std::string str  = std::move(dest).str();
    assert(view == str);
    LIBCPP_ASSERT(str == "o wor");
    assert(dest.str().empty());
    assert(dest.view().empty());
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
