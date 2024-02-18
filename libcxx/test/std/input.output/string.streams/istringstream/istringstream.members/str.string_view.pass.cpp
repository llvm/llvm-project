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

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "nasty_string.h"
#include "test_allocator.h"
#include "test_macros.h"

#include "../../types.h"

template <typename S, typename T>
concept HasStr = requires(S s, const T& sv) {
  { s.str(sv) };
};

template <typename CharT>
void test_sfinae() {
  using SSTREAM  = std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>;
  using CSSTREAM = std::basic_istringstream<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;

  static_assert(HasStr<std::basic_istringstream<CharT>, CharT*>);
  static_assert(HasStr<CSSTREAM, CharT*>);

  static_assert(HasStr<std::basic_istringstream<CharT>, const CharT*>);
  static_assert(HasStr<CSSTREAM, const CharT*>);

  static_assert(HasStr<std::basic_istringstream<CharT>, std::basic_string_view<CharT>>);
  static_assert(HasStr<CSSTREAM, std::basic_string_view<CharT, constexpr_char_traits<CharT>>>);

  static_assert(HasStr<std::basic_istringstream<CharT>, std::basic_string<CharT>>);
  static_assert(HasStr<CSSTREAM, std::basic_string<CharT, constexpr_char_traits<CharT>>>);

  using NSSTREAM = std::basic_istringstream<nasty_char, nasty_char_traits, test_allocator<nasty_char>>;

  static_assert(HasStr<NSSTREAM, nasty_char*>);
  static_assert(HasStr<NSSTREAM, const nasty_char*>);

  static_assert(HasStr<std::basic_istringstream<CharT>, ConstConvertibleStringView<CharT>>);
  static_assert(HasStr<CSSTREAM, ConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>>);

  static_assert(!HasStr<std::basic_istringstream<CharT>, CharT>);
  static_assert(!HasStr<std::basic_istringstream<CharT>, int>);
  static_assert(!HasStr<std::basic_istringstream<CharT>, SomeObject>);
  static_assert(!HasStr<std::basic_istringstream<CharT>, std::nullptr_t>);
  static_assert(!HasStr<std::basic_istringstream<CharT>, NonConstConvertibleStringView<CharT>>);
}

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <typename CharT>
void test() {
  std::basic_istringstream<CharT> ss;
  assert(ss.str().empty());

  // const CharT*
  ss.str(CS("ba"));
  assert(ss.str() == CS("ba"));

  // std::basic_string_view<CharT>
  ss.str(SV("ma"));
  assert(ss.str() == CS("ma"));

  // std::basic_string<CharT>
  ss.str(ST("zmt"));
  assert(ss.str() == CS("zmt"));

  // ConstConvertibleStringView<CharT>
  ss.str(ConstConvertibleStringView<CharT>{CS("da")});
  assert(ss.str() == CS("da"));

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
