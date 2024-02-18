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
  using SSTREAM  = SSTREAM;
  using CSSTREAM = std::basic_istringstream<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;

  // `CharT*`
  static_assert(std::constructible_from<SSTREAM, CharT*, test_allocator<CharT>>);
  static_assert(test_convertible<SSTREAM, CharT*, const test_allocator<CharT>>());

  static_assert(std::constructible_from<CSSTREAM, CharT*, test_allocator<CharT>>);
  static_assert(test_convertible<CSSTREAM, CharT*, const test_allocator<CharT>>());

  // `const CharT*`
  static_assert(std::constructible_from<SSTREAM, const CharT*, const test_allocator<CharT>>);
  static_assert(test_convertible<SSTREAM, const CharT*, const test_allocator<CharT>>());

  static_assert(std::constructible_from<CSSTREAM, const CharT*, const test_allocator<CharT>>);
  static_assert(test_convertible<CSSTREAM, const CharT*, const test_allocator<CharT>>());

  // `std::basic_string_view<CharT>`
  static_assert(std::constructible_from<SSTREAM, const std::basic_string_view<CharT>, const test_allocator<CharT>>);
  static_assert(test_convertible<SSTREAM, std::basic_string_view<CharT>, test_allocator<CharT>>());

  static_assert(std::constructible_from<CSSTREAM, const std::basic_string_view<CharT>, const test_allocator<CharT>>);
  static_assert(test_convertible<CSSTREAM, std::basic_string_view<CharT>, test_allocator<CharT>>());

  // `std::basic_string<CharT>`
  static_assert(std::constructible_from<SSTREAM, const std::basic_string<CharT>, const test_allocator<CharT>>);
  static_assert(test_convertible<SSTREAM, const std::basic_string<CharT>, const test_allocator<CharT>>());
  static_assert(std::constructible_from<CSSTREAM, const std::basic_string<CharT>, const test_allocator<CharT>>);
  static_assert(test_convertible<CSSTREAM, const std::basic_string<CharT>, const test_allocator<CharT>>());

  // nasty_char*
  using NSSTREAM = std::basic_istringstream<nasty_char, nasty_char_traits, test_allocator<nasty_char>>;

  static_assert(std::constructible_from<NSSTREAM, nasty_char*, test_allocator<nasty_char>>);
  static_assert(test_convertible<NSSTREAM, nasty_char*, const test_allocator<nasty_char>>());

  // const nasty_char*
  static_assert(std::constructible_from<NSSTREAM, const nasty_char*, test_allocator<nasty_char>>);
  static_assert(test_convertible<NSSTREAM, const nasty_char*, const test_allocator<nasty_char>>());

  // ConstConvertibleStringView<CharT>
  static_assert(std::constructible_from<SSTREAM,
                                        const ConstConvertibleStringView<CharT>,

                                        const test_allocator<CharT>>);
  static_assert(test_convertible<SSTREAM,
                                 const ConstConvertibleStringView<CharT>,

                                 const test_allocator<CharT>>());

  static_assert(std::constructible_from<CSSTREAM,
                                        const ConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                        const test_allocator<CharT>>);
  static_assert(test_convertible<CSSTREAM,
                                 const ConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                 const test_allocator<CharT>>());

  // NonConstConvertibleStringView<CharT>
  static_assert(!std::constructible_from<SSTREAM, NonConstConvertibleStringView<CharT>, const test_allocator<CharT>>);
  static_assert(!test_convertible<SSTREAM, NonConstConvertibleStringView<CharT>, const test_allocator<CharT>>());

  static_assert(
      !std::constructible_from<SSTREAM, const NonConstConvertibleStringView<CharT>, const test_allocator<CharT>>);
  static_assert(!test_convertible<SSTREAM, const NonConstConvertibleStringView<CharT>, const test_allocator<CharT>>());

  static_assert(!std::constructible_from<CSSTREAM,
                                         const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                         const test_allocator<CharT>>);
  static_assert(!test_convertible<CSSTREAM,
                                  const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                  const test_allocator<CharT>>());

  static_assert(!std::constructible_from<CSSTREAM,
                                         const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                         const test_allocator<CharT>>);
  static_assert(!test_convertible<CSSTREAM,
                                  const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                  const test_allocator<CharT>>());

  // Non-`string-view-like`
  static_assert(!std::constructible_from<SSTREAM, const SomeObject, const test_allocator<CharT>>);
  static_assert(!test_convertible<SSTREAM, const SomeObject, const test_allocator<CharT>>());

  // Non-allocator
  static_assert(!std::constructible_from<SSTREAM, const std::basic_string_view<CharT>, const NonAllocator>);
  static_assert(!test_convertible<SSTREAM, const std::basic_string_view<CharT>, const NonAllocator>());
}

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test() {
  using SSTREAM = SSTREAM;

  const test_allocator<CharT> ca;

  // const CharT*
  {
    SSTREAM ss(CS("zmt"), ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string_view<CharT>
  {
    const std::basic_string_view<CharT> csv = SV("zmt");
    SSTREAM ss(csv, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string<CharT>
  {
    const std::basic_string<CharT> cs = ST("zmt");
    SSTREAM ss(cs, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // ConstConvertibleStringView<CharT>
  {
    const ConstConvertibleStringView<CharT> sv{CS("zmt")};
    SSTREAM ss(sv, ca);
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
