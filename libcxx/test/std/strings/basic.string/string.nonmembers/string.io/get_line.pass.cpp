//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   basic_istream<charT,traits>&
//   getline(basic_istream<charT,traits>& is,
//           basic_string<charT,traits,Allocator>& str);

#include <string>
#include <sstream>
#include <cassert>

#include "min_allocator.h"
#include "test_macros.h"

template <template <class> class Alloc>
void test_string() {
  using S = std::basic_string<char, std::char_traits<char>, Alloc<char> >;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  using WS = std::basic_string<wchar_t, std::char_traits<wchar_t>, Alloc<wchar_t> >;
#endif
  {
    std::istringstream in(" abc\n  def\n   ghij");
    S s("initial text");
    std::getline(in, s);
    assert(in.good());
    assert(s == " abc");
    std::getline(in, s);
    assert(in.good());
    assert(s == "  def");
    std::getline(in, s);
    assert(in.eof());
    assert(s == "   ghij");
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wistringstream in(L" abc\n  def\n   ghij");
    WS s(L"initial text");
    std::getline(in, s);
    assert(in.good());
    assert(s == L" abc");
    std::getline(in, s);
    assert(in.good());
    assert(s == L"  def");
    std::getline(in, s);
    assert(in.eof());
    assert(s == L"   ghij");
  }
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    std::basic_stringbuf<char> sb("hello");
    std::basic_istream<char> is(&sb);
    is.exceptions(std::ios_base::eofbit);

    S s;
    bool threw = false;
    try {
      std::getline(is, s);
    } catch (std::ios::failure const&) {
      threw = true;
    }

    assert(!is.bad());
    assert(!is.fail());
    assert(is.eof());
    assert(threw);
    assert(s == "hello");
  }
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::basic_stringbuf<wchar_t> sb(L"hello");
    std::basic_istream<wchar_t> is(&sb);
    is.exceptions(std::ios_base::eofbit);

    WS s;
    bool threw = false;
    try {
      std::getline(is, s);
    } catch (std::ios::failure const&) {
      threw = true;
    }

    assert(!is.bad());
    assert(!is.fail());
    assert(is.eof());
    assert(threw);
    assert(s == L"hello");
  }
#  endif

  {
    std::basic_stringbuf<char> sb;
    std::basic_istream<char> is(&sb);
    is.exceptions(std::ios_base::failbit);

    S s;
    bool threw = false;
    try {
      std::getline(is, s);
    } catch (std::ios::failure const&) {
      threw = true;
    }

    assert(!is.bad());
    assert(is.fail());
    assert(is.eof());
    assert(threw);
    assert(s == "");
  }
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::basic_stringbuf<wchar_t> sb;
    std::basic_istream<wchar_t> is(&sb);
    is.exceptions(std::ios_base::failbit);

    WS s;
    bool threw = false;
    try {
      std::getline(is, s);
    } catch (std::ios::failure const&) {
      threw = true;
    }

    assert(!is.bad());
    assert(is.fail());
    assert(is.eof());
    assert(threw);
    assert(s == L"");
  }
#  endif
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test_string<std::allocator>();
#if TEST_STD_VER >= 11
  test_string<min_allocator>();
#endif

  return 0;
}
