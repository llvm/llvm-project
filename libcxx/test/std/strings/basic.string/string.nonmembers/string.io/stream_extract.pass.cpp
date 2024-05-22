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
//   operator>>(basic_istream<charT,traits>& is,
//              basic_string<charT,traits,Allocator>& str);

#include <string>
#include <sstream>
#include <cassert>

#include "min_allocator.h"
#include "test_macros.h"

template <template <class> class Alloc>
void test_string() {
  using S = std::basic_string<char, std::char_traits<char>, Alloc<char> >;
  {
    std::istringstream in("a bc defghij");
    S s("initial text");
    in >> s;
    assert(in.good());
    assert(s == "a");
    assert(in.peek() == ' ');
    in >> s;
    assert(in.good());
    assert(s == "bc");
    assert(in.peek() == ' ');
    in.width(3);
    in >> s;
    assert(in.good());
    assert(s == "def");
    assert(in.peek() == 'g');
    in >> s;
    assert(in.eof());
    assert(s == "ghij");
    in >> s;
    assert(in.fail());
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    using WS = std::basic_string<wchar_t, std::char_traits<wchar_t>, Alloc<wchar_t> >;
    std::wistringstream in(L"a bc defghij");
    WS s(L"initial text");
    in >> s;
    assert(in.good());
    assert(s == L"a");
    assert(in.peek() == L' ');
    in >> s;
    assert(in.good());
    assert(s == L"bc");
    assert(in.peek() == L' ');
    in.width(3);
    in >> s;
    assert(in.good());
    assert(s == L"def");
    assert(in.peek() == L'g');
    in >> s;
    assert(in.eof());
    assert(s == L"ghij");
    in >> s;
    assert(in.fail());
  }
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    std::stringbuf sb;
    std::istream is(&sb);
    is.exceptions(std::ios::failbit);

    bool threw = false;
    try {
      S s;
      is >> s;
    } catch (std::ios::failure const&) {
      threw = true;
    }

    assert(!is.bad());
    assert(is.fail());
    assert(is.eof());
    assert(threw);
  }
  {
    std::stringbuf sb;
    std::istream is(&sb);
    is.exceptions(std::ios::eofbit);

    bool threw = false;
    try {
      S s;
      is >> s;
    } catch (std::ios::failure const&) {
      threw = true;
    }

    assert(!is.bad());
    assert(is.fail());
    assert(is.eof());
    assert(threw);
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test_string<std::allocator>();
#if TEST_STD_VER >= 11
  test_string<min_allocator>();
#endif

  return 0;
}
