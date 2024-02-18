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
//   explicit basic_istringstream(const T& t, ios_base::openmode which);

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
  using SSTREAM  = std::basic_istringstream<CharT, std::char_traits<CharT>>;
  using CSSTREAM = std::basic_istringstream<CharT, constexpr_char_traits<CharT>>;

  // `CharT*`

  static_assert(std::constructible_from<SSTREAM, CharT*, std::ios_base::openmode>);
  static_assert(!test_convertible<SSTREAM, CharT*, std::ios_base::openmode>());

  static_assert(std::constructible_from<CSSTREAM, CharT*, std::ios_base::openmode>);
  static_assert(!test_convertible<CSSTREAM, CharT*, std::ios_base::openmode>());

  // `const CharT*`
  static_assert(std::constructible_from<SSTREAM, const CharT*, std::ios_base::openmode>);
  static_assert(!test_convertible<SSTREAM, const CharT*, std::ios_base::openmode>());

  static_assert(std::constructible_from<CSSTREAM, const CharT*, std::ios_base::openmode>);
  static_assert(!test_convertible<CSSTREAM, const CharT*, std::ios_base::openmode, std::ios_base::openmode>());

  // `std::basic_string_view<CharT>`
  static_assert(std::constructible_from<SSTREAM,
                                        const std::basic_string_view<CharT, std::char_traits<CharT>>,
                                        std::ios_base::openmode>);
  static_assert(!test_convertible<SSTREAM,
                                  const std::basic_string_view<CharT, std::char_traits<CharT>>,
                                  std::ios_base::openmode>());

  static_assert(std::constructible_from<CSSTREAM,
                                        const std::basic_string_view<CharT, constexpr_char_traits<CharT>>,
                                        std::ios_base::openmode>);
  static_assert(!test_convertible<CSSTREAM,
                                  const std::basic_string_view<CharT, constexpr_char_traits<CharT>>,
                                  std::ios_base::openmode>());

  // `std::basic_string<CharT>`
  static_assert(std::constructible_from<SSTREAM, const std::basic_string<CharT>, std::ios_base::openmode>);
  static_assert(!test_convertible<SSTREAM, const std::basic_string<CharT>, std::ios_base::openmode>());

  static_assert(std::constructible_from<CSSTREAM,
                                        const std::basic_string<CharT, constexpr_char_traits<CharT>>,
                                        std::ios_base::openmode>);
  static_assert(!test_convertible<CSSTREAM,
                                  const std::basic_string<CharT, constexpr_char_traits<CharT>>,
                                  std::ios_base::openmode>());

  // ConstConvertibleStringView<CharT>
  static_assert(std::constructible_from<SSTREAM, const ConstConvertibleStringView<CharT>, std::ios_base::openmode>);
  static_assert(!test_convertible<SSTREAM, const ConstConvertibleStringView<CharT>, std::ios_base::openmode>());

  static_assert(std::constructible_from<CSSTREAM,
                                        const ConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                        std::ios_base::openmode>);
  static_assert(!test_convertible<CSSTREAM,
                                  const ConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                  std::ios_base::openmode>());

  // NonConstConvertibleStringView<CharT>
  static_assert(!std::constructible_from<SSTREAM, NonConstConvertibleStringView<CharT>, std::ios_base::openmode>);
  static_assert(!test_convertible<SSTREAM, NonConstConvertibleStringView<CharT>, std::ios_base::openmode>());

  static_assert(!std::constructible_from<SSTREAM, const NonConstConvertibleStringView<CharT>, std::ios_base::openmode>);
  static_assert(!test_convertible<SSTREAM, const NonConstConvertibleStringView<CharT>, std::ios_base::openmode>());

  static_assert(!std::constructible_from<CSSTREAM,
                                         const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                         std::ios_base::openmode>);
  static_assert(!test_convertible<CSSTREAM,
                                  const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                  std::ios_base::openmode>());

  static_assert(!std::constructible_from<CSSTREAM,
                                         const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                         std::ios_base::openmode>);
  static_assert(!test_convertible<CSSTREAM,
                                  const NonConstConvertibleStringView<CharT, constexpr_char_traits<CharT>>,
                                  std::ios_base::openmode>());

  //  nasty_char*
  using NSSTREAM = std::basic_istringstream<nasty_char, nasty_char_traits, test_allocator<nasty_char>>;

  static_assert(std::constructible_from<NSSTREAM, nasty_char*, test_allocator<nasty_char>>);
  static_assert(!test_convertible<NSSTREAM, nasty_char*, std::ios_base::openmode>());

  // const nasty_char*
  using NSSTREAM = std::basic_istringstream<nasty_char, nasty_char_traits, test_allocator<nasty_char>>;

  static_assert(std::constructible_from<NSSTREAM, const nasty_char*, test_allocator<nasty_char>>);
  static_assert(!test_convertible<NSSTREAM, const nasty_char*, std::ios_base::openmode>());

  // Non-`string-view-like`
  static_assert(!std::constructible_from<SSTREAM, const SomeObject, std::ios_base::openmode>);
  static_assert(!test_convertible<SSTREAM, const SomeObject, std::ios_base::openmode>());

  // Non-mode
  static_assert(!std::constructible_from<SSTREAM, const std::basic_string_view<CharT>, const SomeObject>);
  static_assert(!test_convertible<SSTREAM, const std::basic_string_view<CharT>, const SomeObject>());
}

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test() {
  using SSTREAM = std::basic_istringstream<CharT, std::char_traits<CharT>>;

  // const CharT*
  {
    SSTREAM ss(CS("zmt"), std::ios_base::binary);
    assert(ss.str() == CS("zmt"));
  }
  // std::basic_string_view<CharT>
  {
    const std::basic_string_view<CharT> csv = SV("zmt");
    SSTREAM ss(csv, std::ios_base::binary);
    assert(ss.str() == CS("zmt"));
  }
  // std::basic_string<CharT>
  {
    const std::basic_string<CharT> cs = ST("zmt");
    SSTREAM ss(cs, std::ios_base::binary);
    assert(ss.str() == CS("zmt"));
  }
  // ConstConvertibleStringView<CharT>
  {
    const ConstConvertibleStringView<CharT> sv{CS("zmt")};
    SSTREAM ss(sv, std::ios_base::binary);
    assert(ss.str() == CS("zmt"));
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
