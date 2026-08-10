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
//   basic_stringbuf(const T& t, ios_base::openmode which, const Allocator& a);

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
  using NStrBuf = std::basic_stringbuf<nasty_char, nasty_char_traits, test_allocator<nasty_char>>;

  static_assert(std::constructible_from<NStrBuf, nasty_char*, test_allocator<nasty_char>>);
  static_assert(test_convertible<NStrBuf, nasty_char*, std::ios_base::openmode, const test_allocator<nasty_char>>());

  // const nasty_char*
  static_assert(std::constructible_from<NStrBuf, const nasty_char*, test_allocator<nasty_char>>);
  static_assert(
      test_convertible<NStrBuf, const nasty_char*, std::ios_base::openmode, const test_allocator<nasty_char>>());
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>, typename AllocT = std::allocator<CharT>>
void test_sfinae() {
  using StrBuf = std::basic_stringbuf<CharT, TraitsT, AllocT>;

  // `CharT*`
  static_assert(std::constructible_from<StrBuf, CharT*, AllocT>);
  static_assert(test_convertible<StrBuf, CharT*, std::ios_base::openmode, const AllocT>());

  // `const CharT*`
  static_assert(std::constructible_from<StrBuf, const CharT*, AllocT>);
  static_assert(test_convertible<StrBuf, const CharT*, std::ios_base::openmode, const AllocT>());

  // `std::basic_string_view<CharT>`
  static_assert(std::constructible_from<StrBuf,
                                        const std::basic_string_view<CharT, TraitsT>,
                                        std::ios_base::openmode,
                                        const AllocT>);
  static_assert(
      test_convertible<StrBuf, const std::basic_string_view<CharT, TraitsT>, std::ios_base::openmode, const AllocT>());

  // `std::basic_string<CharT>`
  static_assert(
      std::constructible_from<StrBuf, const std::basic_string<CharT, TraitsT>, std::ios_base::openmode, const AllocT>);
  static_assert(
      test_convertible<StrBuf, const std::basic_string<CharT, TraitsT>, std::ios_base::openmode, const AllocT>());

  // ConstConvertibleStringView<CharT>
  static_assert(std::constructible_from<StrBuf,
                                        const ConstConvertibleStringView<CharT, TraitsT>,
                                        std::ios_base::openmode,
                                        const AllocT>);
  static_assert(test_convertible<StrBuf,
                                 const ConstConvertibleStringView<CharT, TraitsT>,
                                 std::ios_base::openmode,
                                 const AllocT>());

  // NonConstConvertibleStringView<CharT>
  static_assert(!std::constructible_from<StrBuf,
                                         NonConstConvertibleStringView<CharT, TraitsT>,
                                         std::ios_base::openmode,
                                         const AllocT>);
  static_assert(!test_convertible<StrBuf,
                                  NonConstConvertibleStringView<CharT, TraitsT>,
                                  std::ios_base::openmode,
                                  const AllocT>());

  static_assert(!std::constructible_from<StrBuf,
                                         const NonConstConvertibleStringView<CharT, TraitsT>,
                                         std::ios_base::openmode,
                                         const AllocT>);
  static_assert(!test_convertible<StrBuf,
                                  const NonConstConvertibleStringView<CharT, TraitsT>,
                                  std::ios_base::openmode,
                                  const AllocT>());

  // Non-`string-view-like`
  static_assert(!std::constructible_from<StrBuf, const SomeObject, std::ios_base::openmode, const AllocT>);
  static_assert(!test_convertible<StrBuf, const SomeObject, std::ios_base::openmode, const AllocT>());

  static_assert(!std::constructible_from<StrBuf, const int, std::ios_base::openmode, const AllocT>);
  static_assert(!test_convertible<StrBuf, const int, std::ios_base::openmode, const AllocT>());

  // Non-mode
  static_assert(
      !std::constructible_from<StrBuf, const std::basic_string_view<CharT, TraitsT>, NonMode, const NonAllocator>);
  static_assert(!test_convertible<StrBuf, const std::basic_string_view<CharT, TraitsT>, NonMode, const NonAllocator>());

  // Non-allocator
  static_assert(!std::constructible_from<StrBuf,
                                         const std::basic_string_view<CharT, TraitsT>,
                                         std::ios_base::openmode,
                                         const NonAllocator>);
  static_assert(!test_convertible<StrBuf,
                                  const std::basic_string_view<CharT, TraitsT>,
                                  std::ios_base::openmode,
                                  const NonAllocator>());
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>, typename AllocT = std::allocator<CharT>>
void test() {
  using StrBuf = std::basic_stringbuf<CharT, TraitsT, AllocT>;

  const AllocT allocator;

  // const CharT*
  {
    StrBuf ss(CS("zmt"), std::ios_base::out | std::ios_base::in, allocator);
    assert(ss.str() == CS("zmt"));
    assert(ss.get_allocator() == allocator);
  }
  // std::basic_string_view<CharT>
  {
    const std::basic_string_view<CharT, TraitsT> csv = SV("zmt");
    StrBuf ss(csv, std::ios_base::out | std::ios_base::in, allocator);
    assert(ss.str() == CS("zmt"));
    assert(ss.get_allocator() == allocator);
  }
  // std::basic_string<CharT>
  {
    const std::basic_string<CharT, TraitsT, AllocT> cs = ST("zmt", allocator);
    StrBuf ss(cs, std::ios_base::out | std::ios_base::in, allocator);
    assert(ss.str() == CS("zmt"));
    assert(ss.get_allocator() == allocator);
  }
  // ConstConvertibleStringView<CharT>
  {
    const ConstConvertibleStringView<CharT, TraitsT> sv{CS("zmt")};
    StrBuf ss(sv, std::ios_base::out | std::ios_base::in, allocator);
    assert(ss.str() == CS("zmt"));
    assert(ss.get_allocator() == allocator);
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
