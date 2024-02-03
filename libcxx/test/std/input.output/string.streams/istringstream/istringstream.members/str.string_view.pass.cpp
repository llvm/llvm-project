//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT>>
// class basic_istringstream

// template<class T>
//   void str(const T& t);

#include <cassert>
#include <sstream>
#include <string>
#include <string_view>

#include "make_string.h"
#include "test_macros.h"

template <typename S, typename T>
concept HasStr = requires(S s, const T& sv) {
  { s.str(sv) };
};

template <typename CharT>
void test_sfinae() {
  struct SomeObject {};

  static_assert(HasStr<std::basic_istringstream<CharT>, const CharT*>);
  static_assert(HasStr<std::basic_istringstream<CharT>, std::basic_string_view<CharT>>);
  static_assert(HasStr<std::basic_istringstream<CharT>, std::basic_string<CharT>>);

  static_assert(!HasStr<std::basic_istringstream<CharT>, char>);
  static_assert(!HasStr<std::basic_istringstream<CharT>, int>);
  static_assert(!HasStr<std::basic_istringstream<CharT>, SomeObject>);
  static_assert(!HasStr<std::basic_istringstream<CharT>, std::nullptr_t>);
}

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <typename CharT>
void test() {
  std::basic_istringstream<CharT> ss;
  assert(ss.str().empty());
  ss.str(CS("ba"));
  assert(ss.str() == CS("ba"));
  ss.str(SV("ma"));
  assert(ss.str() == CS("ma"));
  ss.str(ST("zmt"));
  assert(ss.str() == CS("zmt"));
  const std::basic_string<CharT> s;
  ss.str(s);
  assert(ss.str().empty());
}

int main(int, char**) {
  test_sfinae<char>();
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sfinae<wchar_t>();
  test<wchar_t>();
#endif

  return 0;
}
