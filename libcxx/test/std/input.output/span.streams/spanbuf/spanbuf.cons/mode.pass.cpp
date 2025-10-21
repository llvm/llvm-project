//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

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

#include "../../helper_types.h"

template <typename CharT, typename TraitsT>
void test_sfinae() {
  using SpanBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Mode
  static_assert(std::constructible_from<SpanBuf, std::ios_base::openmode>);
  static_assert(!test_convertible<SpanBuf, std::ios_base::openmode>());

  // Non-mode
  struct NonMode {};
  static_assert(!std::constructible_from<SpanBuf, const NonMode>);
  static_assert(!test_convertible<SpanBuf, const NonMode>());
}

template <typename CharT, typename TraitsT>
void test() {
  using SpanBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Mode: `in`
  {
    SpanBuf spanBuf{std::ios_base::in};
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);
  }
  // Mode: `out`
  {
    SpanBuf spanBuf{std::ios_base::out};
    assert(spanBuf.span().data() == nullptr);
    // Mode `out` counts read characters
    assert(spanBuf.span().size() == 0);
  }
  // Mode: `ate`
  {
    SpanBuf spanBuf{std::ios_base::ate};
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);
  }
  // Mode: `ate`
  {
    SpanBuf spanBuf{std::ios_base::out | std::ios_base::ate};
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  test_sfinae<nasty_char, nasty_char_traits>();
#endif
  test_sfinae<char, constexpr_char_traits<char>>();
  test_sfinae<char, std::char_traits<char>>();

#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_char, nasty_char_traits>();
#endif
  test<char, constexpr_char_traits<char>>();
  test<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sfinae<wchar_t, constexpr_char_traits<wchar_t>>();
  test_sfinae<wchar_t, std::char_traits<wchar_t>>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
  test<wchar_t, std::char_traits<wchar_t>>();
#endif

  return 0;
}
