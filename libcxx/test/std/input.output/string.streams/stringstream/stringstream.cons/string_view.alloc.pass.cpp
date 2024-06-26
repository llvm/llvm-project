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
// class basic_stringstream

// template<class T>
//   basic_stringstream(const T& t, const Allocator& a);

#include <cassert>
#include <concepts>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_allocator.h"
#include "test_convertible.h"
#include "test_macros.h"

#include "../../helper_string_macros.h"
#include "../../helper_types.h"

template <typename AllocT = std::allocator<nasty_char>>
void test_sfinae_with_nasty_char() {
  // nasty_char*
  using NStrStream = std::basic_stringstream<nasty_char, nasty_char_traits, AllocT>;

  static_assert(std::constructible_from<NStrStream, nasty_char*, AllocT>);
  static_assert(test_convertible<NStrStream, nasty_char*, const AllocT>());

  // const nasty_char*
  static_assert(std::constructible_from<NStrStream, const nasty_char*, AllocT>);
  static_assert(test_convertible<NStrStream, const nasty_char*, const AllocT>());
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>, typename AllocT = std::allocator<CharT>>
void test_sfinae() {
  using StrStream = std::basic_stringstream<CharT, TraitsT, AllocT>;

  // `CharT*`
  static_assert(std::constructible_from<StrStream, CharT*, const AllocT>);
  static_assert(test_convertible<StrStream, CharT*, const AllocT>());

  // `const CharT*`
  static_assert(std::constructible_from<StrStream, const CharT*, const AllocT>);
  static_assert(test_convertible<StrStream, const CharT*, const AllocT>());

  // `std::basic_string_view<CharT>`
  static_assert(std::constructible_from<StrStream, const std::basic_string_view<CharT, TraitsT>, const AllocT>);
  static_assert(test_convertible<StrStream, std::basic_string_view<CharT, TraitsT>, const AllocT>());

  // `std::basic_string<CharT>`
  static_assert(std::constructible_from<StrStream, const std::basic_string<CharT, TraitsT>, const AllocT>);
  static_assert(test_convertible<StrStream, const std::basic_string<CharT, TraitsT>, const AllocT>());

  // ConstConvertibleStringView<CharT>
  static_assert(std::constructible_from<StrStream, const ConstConvertibleStringView<CharT, TraitsT>, const AllocT>);
  static_assert(test_convertible<StrStream, const ConstConvertibleStringView<CharT, TraitsT>, const AllocT>());

  // NonConstConvertibleStringView<CharT>
  static_assert(!std::constructible_from<StrStream, NonConstConvertibleStringView<CharT, TraitsT>, const AllocT>);
  static_assert(!test_convertible<StrStream, NonConstConvertibleStringView<CharT, TraitsT>, const AllocT>());

  static_assert(!std::constructible_from<StrStream, const NonConstConvertibleStringView<CharT, TraitsT>, const AllocT>);
  static_assert(!test_convertible<StrStream, const NonConstConvertibleStringView<CharT, TraitsT>, const AllocT>());

  // Non-`string-view-like`
  static_assert(!std::constructible_from<StrStream, const SomeObject, const AllocT>);
  static_assert(!test_convertible<StrStream, const SomeObject, const AllocT>());

  // Non-allocator
  static_assert(!std::constructible_from<StrStream, const std::basic_string_view<CharT, TraitsT>, const NonAllocator>);
  static_assert(!test_convertible<StrStream, const std::basic_string_view<CharT, TraitsT>, const NonAllocator>());
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>, typename AllocT = std::allocator<CharT>>
void test() {
  using StrStream = std::basic_stringstream<CharT, TraitsT, AllocT>;

  const AllocT allocator;

  // const CharT*
  {
    StrStream ss(CS("zmt"), allocator);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == allocator);
  }
  // std::basic_string_view<CharT>
  {
    const std::basic_string_view<CharT, TraitsT> csv = SV("zmt");
    StrStream ss(csv, allocator);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == allocator);
  }
  // std::basic_string<CharT>
  {
    const std::basic_string<CharT, TraitsT, AllocT> cs = ST("zmt", allocator);
    StrStream ss(cs, allocator);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == allocator);
  }
  // ConstConvertibleStringView<CharT>
  {
    const ConstConvertibleStringView<CharT, TraitsT> sv{CS("zmt")};
    StrStream ss(sv, allocator);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == allocator);
  }
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
