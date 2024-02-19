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
//   basic_istringstream(const T& t, const Allocator& a);

#include <cassert>
#include <concepts>
#include <sstream>
#include <string>
#include <string_view>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "nasty_string.h"
#include "test_allocator.h"
#include "test_convertible.h"
#include "test_macros.h"

#include "../../types.h"

template <typename CharT>
void test_sfinae() {
  using StrStream  = std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>;
  using CStrStream = std::basic_istringstream<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;

  // `CharT*`
  static_assert(std::constructible_from<StrStream, CharT*, const test_allocator<CharT>>);
  static_assert(test_convertible<StrStream, CharT*, const test_allocator<CharT>>());

  static_assert(std::constructible_from<CStrStream, CharT*, const test_allocator<CharT>>);
  static_assert(test_convertible<CStrStream, CharT*, const test_allocator<CharT>>());

  // `const CharT*`
  static_assert(std::constructible_from<StrStream, const CharT*, const test_allocator<CharT>>);
  static_assert(test_convertible<StrStream, const CharT*, const test_allocator<CharT>>());

  static_assert(std::constructible_from<CStrStream, const CharT*, const test_allocator<CharT>>);
  static_assert(test_convertible<CStrStream, const CharT*, const test_allocator<CharT>>());

  // `std::basic_string_view<CharT>`
  static_assert(std::constructible_from<StrStream, const std::basic_string_view<CharT>, const test_allocator<CharT>>);
  static_assert(test_convertible<StrStream, std::basic_string_view<CharT>, const test_allocator<CharT>>());

  static_assert(std::constructible_from<CStrStream,
                                        const std::basic_string_view<CharT, constexpr_char_traits<CharT>>,
                                        const test_allocator<CharT>>);
  static_assert(test_convertible<CStrStream,
                                 std::basic_string_view<CharT, constexpr_char_traits<CharT>>,
                                 const test_allocator<CharT>>());

  // `std::basic_string<CharT>`
  static_assert(std::constructible_from<StrStream, const std::basic_string<CharT>, const test_allocator<CharT>>);
  static_assert(test_convertible<StrStream, const std::basic_string<CharT>, const test_allocator<CharT>>());

  static_assert(std::constructible_from<CStrStream,
                                        const std::basic_string<CharT, constexpr_char_traits<CharT>>,
                                        const test_allocator<CharT>>);
  static_assert(test_convertible<CStrStream,
                                 const std::basic_string<CharT, constexpr_char_traits<CharT>>,
                                 const test_allocator<CharT>>());

  // ConstConvertibleStringView<CharT>
  static_assert(std::constructible_from<StrStream, const ConstConvertibleStringView<CharT>, const test_allocator<CharT>>);
  static_assert(test_convertible<StrStream, const ConstConvertibleStringView<CharT>, const test_allocator<CharT>>());

  static_assert(std::constructible_from<CStrStream,
                                        const ConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                        const test_allocator<CharT>>);
  static_assert(test_convertible<CStrStream,
                                 const ConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                 const test_allocator<CharT>>());

  // NonConstConvertibleStringView<CharT>
  static_assert(!std::constructible_from<StrStream, NonConstConvertibleStringView<CharT>, const test_allocator<CharT>>);
  static_assert(!test_convertible<StrStream, NonConstConvertibleStringView<CharT>, const test_allocator<CharT>>());

  static_assert(
      !std::constructible_from<StrStream, const NonConstConvertibleStringView<CharT>, const test_allocator<CharT>>);
  static_assert(!test_convertible<StrStream, const NonConstConvertibleStringView<CharT>, const test_allocator<CharT>>());

  static_assert(!std::constructible_from<CStrStream,
                                         const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                         const test_allocator<CharT>>);
  static_assert(!test_convertible<CStrStream,
                                  const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                  const test_allocator<CharT>>());

  static_assert(!std::constructible_from<CStrStream,
                                         const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                         const test_allocator<CharT>>);
  static_assert(!test_convertible<CStrStream,
                                  const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                  const test_allocator<CharT>>());

  // nasty_char*
  using NStrStream = std::basic_istringstream<nasty_char, nasty_char_traits, test_allocator<nasty_char>>;

  static_assert(std::constructible_from<NStrStream, nasty_char*, test_allocator<nasty_char>>);
  static_assert(test_convertible<NStrStream, nasty_char*, const test_allocator<nasty_char>>());

  // const nasty_char*
  static_assert(std::constructible_from<NStrStream, const nasty_char*, test_allocator<nasty_char>>);
  static_assert(test_convertible<NStrStream, const nasty_char*, const test_allocator<nasty_char>>());

  // Non-`string-view-like`
  static_assert(!std::constructible_from<StrStream, const SomeObject, const test_allocator<CharT>>);
  static_assert(!test_convertible<StrStream, const SomeObject, const test_allocator<CharT>>());

  // Non-allocator
  static_assert(!std::constructible_from<StrStream, const std::basic_string_view<CharT>, const NonAllocator>);
  static_assert(!test_convertible<StrStream, const std::basic_string_view<CharT>, const NonAllocator>());
}

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test() {
  using StrStream = std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>;

  const test_allocator<CharT> ca;

  // const CharT*
  {
    StrStream ss(CS("zmt"), ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string_view<CharT>
  {
    const std::basic_string_view<CharT> csv = SV("zmt");
    StrStream ss(csv, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string<CharT>
  {
    const std::basic_string<CharT> cs = ST("zmt");
    StrStream ss(cs, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // ConstConvertibleStringView<CharT>
  {
    const ConstConvertibleStringView<CharT> sv{CS("zmt")};
    StrStream ss(sv, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
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
