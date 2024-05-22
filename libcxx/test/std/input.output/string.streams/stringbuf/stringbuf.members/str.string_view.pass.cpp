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
// class basic_stringbuf

// template<class T>
//   void str(const T& t);

#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_allocator.h"
#include "test_macros.h"

#include "../../helper_concepts.h"
#include "../../helper_string_macros.h"
#include "../../helper_types.h"

template <typename AllocT = std::allocator<nasty_char>>
void test_sfinae_with_nasty_char() {
  using NStrBuf = std::basic_stringbuf<nasty_char, nasty_char_traits, AllocT>;

  static_assert(is_valid_argument_for_str_member<NStrBuf, nasty_char*>);
  static_assert(is_valid_argument_for_str_member<NStrBuf, const nasty_char*>);
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>, typename AllocT = std::allocator<CharT>>
void test_sfinae() {
  using StrBuff = std::basic_stringbuf<CharT, TraitsT, AllocT>;

  static_assert(is_valid_argument_for_str_member<StrBuff, CharT*>);
  static_assert(is_valid_argument_for_str_member<StrBuff, const CharT*>);
  static_assert(is_valid_argument_for_str_member<StrBuff, std::basic_string_view<CharT, TraitsT>>);
  static_assert(is_valid_argument_for_str_member<StrBuff, std::basic_string<CharT, TraitsT, AllocT>>);
  static_assert(is_valid_argument_for_str_member<StrBuff, ConstConvertibleStringView<CharT, TraitsT>>);

  static_assert(!is_valid_argument_for_str_member<StrBuff, CharT>);
  static_assert(!is_valid_argument_for_str_member<StrBuff, int>);
  static_assert(!is_valid_argument_for_str_member<StrBuff, SomeObject>);
  static_assert(!is_valid_argument_for_str_member<StrBuff, std::nullptr_t>);
  static_assert(!is_valid_argument_for_str_member<StrBuff, NonConstConvertibleStringView<CharT, TraitsT>>);
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>, typename AllocT = std::allocator<CharT>>
void test() {
  AllocT allocator;

  std::basic_stringbuf<CharT, TraitsT, AllocT> ss(std::ios_base::in | std::ios_base::out, allocator);
  assert(ss.str().empty());

  // const CharT*
  ss.str(CS("ba"));
  assert(ss.str() == CS("ba"));

  // std::basic_string_view<CharT>
  ss.str(SV("ma"));
  assert(ss.str() == CS("ma"));

  // std::basic_string<CharT>
  ss.str(ST("zmt", allocator));
  assert(ss.str() == CS("zmt"));

  // ConstConvertibleStringView<CharT>
  ss.str(ConstConvertibleStringView<CharT, TraitsT>{CS("da")});
  assert(ss.str() == CS("da"));

  const std::basic_string<CharT, TraitsT, AllocT> s(allocator);
  ss.str(s);
  assert(ss.str().empty());
}

int main(int, char**) {
  test_sfinae_with_nasty_char();
  test_sfinae_with_nasty_char<test_allocator<nasty_char>>();
  test_sfinae<char>();
  test_sfinae<char, constexpr_char_traits<char>, std::allocator<char>>();
  test_sfinae<char, std::char_traits<char>, test_allocator<char>>();
  test_sfinae<char, constexpr_char_traits<char>, test_allocator<char>>();
  test<char>();
  test<char, constexpr_char_traits<char>, std::allocator<char>>();
  test<char, std::char_traits<char>, test_allocator<char>>();
  test<char, constexpr_char_traits<char>, test_allocator<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sfinae<wchar_t>();
  test_sfinae<wchar_t, constexpr_char_traits<wchar_t>, std::allocator<wchar_t>>();
  test_sfinae<wchar_t, std::char_traits<wchar_t>, test_allocator<wchar_t>>();
  test_sfinae<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>();
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>, std::allocator<wchar_t>>();
  test<wchar_t, std::char_traits<wchar_t>, test_allocator<wchar_t>>();
  test<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>();
#endif

  return 0;
}
