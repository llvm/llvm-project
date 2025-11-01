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
//   class basic_ospanstream
//     : public basic_ostream<charT, traits> {

//     // [spanstream.cons], constructors
//     explicit basic_ospanstream(std::span<charT> s,
//                                ios_base::openmode which = ios_base::out | ios_base::in);

#include <cassert>
#include <concepts>
#include <span>
#include <spanstream>
#include <utility>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_convertible.h"
#include "test_macros.h"

#include "../../helper_types.h"

template <typename CharT, typename TraitsT>
void test_sfinae() {
  using SpStream = std::basic_ospanstream<CharT, TraitsT>;

  // Mode
  static_assert(std::constructible_from<SpStream, const std::span<CharT>, std::ios_base::openmode>);
  static_assert(!test_convertible<SpStream, const std::span<CharT>, std::ios_base::openmode>());

  // Non-mode
  struct NonMode {};
  static_assert(!std::constructible_from<SpStream, const std::span<CharT>, const NonMode>);
  static_assert(!test_convertible<SpStream, const std::span<CharT>, const NonMode>());
}

template <typename CharT, typename TraitsT>
void test() {
  using SpStream = std::basic_ospanstream<CharT, TraitsT>;

  CharT arr[4];

  std::span<CharT> sp{arr};
  assert(sp.data() == arr);
  assert(sp.size() == 4);

  // Mode: default (`out`)
  {
    SpStream spSt{sp};
    assert(spSt.span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.span().size() == 0);
  }
  {
    SpStream spSt{std::as_const(sp)};
    assert(spSt.span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.span().size() == 0);
  }
  // Mode `out`
  {
    SpStream spSt{sp, std::ios_base::out};
    assert(spSt.span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.span().size() == 0);
  }
  {
    SpStream spSt{std::as_const(sp), std::ios_base::out};
    assert(spSt.span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.span().size() == 0);
  }
  // Mode `ate`
  {
    SpStream spSt{sp, std::ios_base::ate};
    assert(spSt.span().data() == arr);
    assert(spSt.span().size() == 4);
  }
  {
    SpStream spSt{std::as_const(sp), std::ios_base::ate};
    assert(spSt.span().data() == arr);
    assert(spSt.span().size() == 4);
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  test_sfinae<nasty_char, nasty_char_traits>();
#endif

  test_sfinae<char, constexpr_char_traits<char>>();
  test_sfinae<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sfinae<wchar_t, constexpr_char_traits<wchar_t>>();
  test_sfinae<wchar_t, std::char_traits<wchar_t>>();
#endif

#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_char, nasty_char_traits>();
#endif

  test<char, constexpr_char_traits<char>>();
  test<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, constexpr_char_traits<wchar_t>>();
  test<wchar_t, std::char_traits<wchar_t>>();
#endif

  return 0;
}
