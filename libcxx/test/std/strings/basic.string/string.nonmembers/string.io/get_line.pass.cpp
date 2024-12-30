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

#include <cassert>
#include <sstream>
#include <string>

#include "make_string.h"
#include "min_allocator.h"
#include "stream_types.h"
#include "test_macros.h"

template <class CharT, class Alloc, class Stream, class Streambuf>
void test() {
  using string_type    = std::basic_string<CharT, std::char_traits<CharT>, Alloc>;
  using stream_type    = std::basic_istream<CharT>;
  using streambuf_type = Streambuf;

  {
    streambuf_type sb(MAKE_CSTRING(CharT, " abc\n  def\n   ghij"));
    stream_type in(&sb);
    string_type s(MAKE_CSTRING(CharT, "initial text"));
    std::getline(in, s);
    assert(in.good());
    assert(s == MAKE_CSTRING(CharT, " abc"));
    std::getline(in, s);
    assert(in.good());
    assert(s == MAKE_CSTRING(CharT, "  def"));
    std::getline(in, s);
    assert(in.eof());
    assert(s == MAKE_CSTRING(CharT, "   ghij"));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    streambuf_type sb(MAKE_CSTRING(CharT, "hello"));
    stream_type is(&sb);
    is.exceptions(std::ios_base::eofbit);

    string_type s;
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
    assert(s == MAKE_CSTRING(CharT, "hello"));
  }
  {
    streambuf_type sb(MAKE_CSTRING(CharT, ""));
    stream_type is(&sb);
    is.exceptions(std::ios_base::failbit);

    string_type s;
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
    assert(s == MAKE_CSTRING(CharT, ""));
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

template <template <class> class Alloc>
void test_alloc() {
  test<char, Alloc<char>, std::basic_istringstream<char>, std::basic_stringbuf<char> >();
  test<char, Alloc<char>, std::basic_istringstream<char>, non_buffering_streambuf<char> >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, Alloc<wchar_t>, std::basic_istringstream<wchar_t>, std::basic_stringbuf<wchar_t> >();
  test<wchar_t, Alloc<wchar_t>, std::basic_istringstream<wchar_t>, non_buffering_streambuf<wchar_t> >();
#endif
}
/*
void test_tiny_allocator() {
  {
    std::string in_str =
        "this is a too long line for the string that has to be longer because the implementation is broken\n";
    std::istringstream iss(in_str);
    std::basic_string<char, std::char_traits<char>, tiny_size_allocator<40, char>> str;
    std::getline(iss, str);
    assert(str == std::string_view{in_str}.substr(0, str.max_size()));
    assert(iss.rdstate() & std::ios::failbit);
  }
  {
    std::string in_str =
        "this is a too long line for the string that has to be longer because the implementation is broken";
    std::istringstream iss(in_str);
    std::basic_string<char, std::char_traits<char>, tiny_size_allocator<40, char>> str;
    std::getline(iss, str);
    assert(str == std::string_view{in_str}.substr(0, str.max_size()));
    assert(iss.rdstate() & std::ios::failbit);
  }
}
*/

int main(int, char**) {
  test_alloc<std::allocator>();
#if TEST_STD_VER >= 11
  test_alloc<min_allocator>();
#endif
  // test_tiny_allocator();

  return 0;
}
