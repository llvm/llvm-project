//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf

// void str(const basic_string<charT,traits,Allocator>& s);

#include <sstream>
#include <string>
#include <cassert>

#include "test_macros.h"

struct StringBuf : std::stringbuf {
  explicit StringBuf(const char* s, std::ios_base::openmode mode) : basic_stringbuf(s, mode) {}
  void public_setg(int a, int b, int c) {
    char* p = eback();
    this->setg(p + a, p + b, p + c);
  }
};

static void test_altered_sequence_pointers() {
  {
    StringBuf src("hello world", std::ios_base::in);
    src.public_setg(4, 6, 9);
    std::stringbuf dest;
    dest            = std::move(src);
    std::string str = dest.str();
    assert(5 <= str.size() && str.size() <= 11);
    LIBCPP_ASSERT(str == "o wor");
    LIBCPP_ASSERT(dest.str() == "o wor");
  }
  {
    StringBuf src("hello world", std::ios_base::in);
    src.public_setg(4, 6, 9);
    std::stringbuf dest;
    dest.swap(src);
    std::string str = dest.str();
    assert(5 <= str.size() && str.size() <= 11);
    LIBCPP_ASSERT(str == "o wor");
    LIBCPP_ASSERT(dest.str() == "o wor");
  }
}

int main(int, char**)
{
  test_altered_sequence_pointers();
  {
    std::stringbuf buf("testing");
    assert(buf.str() == "testing");
    buf.str("another test");
    assert(buf.str() == "another test");
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wstringbuf buf(L"testing");
        assert(buf.str() == L"testing");
        buf.str(L"another test");
        assert(buf.str() == L"another test");
    }
#endif

  return 0;
}
