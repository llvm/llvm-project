//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <spanstream>

//   template<class charT, class traits = char_traits<charT>>
//   class basic_spanbuf
//     : public basic_streambuf<charT, traits> {

//     // [spanbuf.cons], constructors
//
//     explicit basic_spanbuf(ios_base::openmode which);

#include <cassert>
#include <concepts>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_convertible.h"
#include "test_macros.h"

#include "../../types.h"

void test_sfinae_with_nasty_char() {
  using SpBuf = std::basic_spanbuf<nasty_char, nasty_char_traits>;

  static_assert(std::constructible_from<SpBuf, std::ios_base::openmode>);
  static_assert(!test_convertible<SpBuf, std::ios_base::openmode>());
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test_sfinae() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  // `Mode`
  static_assert(std::constructible_from<SpBuf, std::ios_base::openmode>);
  static_assert(!test_convertible<SpBuf, std::ios_base::openmode>());

  // Non-mode
  static_assert(!std::constructible_from<SpBuf, const NonMode>);
  static_assert(!test_convertible<SpBuf, const NonMode>());
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  static_assert(std::default_initializable<SpBuf>);

  // Mode: `in`
  {
    SpBuf spBuf(std::ios_base::in);
    assert(spBuf.span().data() == nullptr);
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
  }
  // Mode: `out`
  {
    SpBuf spBuf(std::ios_base::out);
    assert(spBuf.span().data() == nullptr);
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
  }
  // Mode: multiple
  {
    SpBuf spBuf(std::ios_base::out | std::ios_base::in | std::ios_base::binary);
    assert(spBuf.span().data() == nullptr);
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
  }
}

int main(int, char**) {
  test_sfinae_with_nasty_char();
  test_sfinae<char>();
  test_sfinae<char, constexpr_char_traits<char>>();
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sfinae<wchar_t>();
  test_sfinae<wchar_t, constexpr_char_traits<wchar_t>>();
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
