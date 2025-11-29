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
//   public:
//     using char_type   = charT;
//     using int_type    = typename traits::int_type;
//     using pos_type    = typename traits::pos_type;
//     using off_type    = typename traits::off_type;
//     using traits_type = traits;

//   using spanbuf = basic_spanbufchar>;
//   using wspanbuf = basic_spanbuf<wchar_t>;

#include <spanstream>
#include <string>
#include <type_traits>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT>
void test() {
  using SpanBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Constructors

  static_assert(std::is_default_constructible_v<SpanBuf>);

  // Types

  static_assert(std::is_base_of_v<std::basic_streambuf<CharT, TraitsT>, SpanBuf>);
  static_assert(std::is_same_v<typename SpanBuf::char_type, CharT>);
  static_assert(std::is_same_v<typename SpanBuf::int_type, typename TraitsT::int_type>);
  static_assert(std::is_same_v<typename SpanBuf::pos_type, typename TraitsT::pos_type>);
  static_assert(std::is_same_v<typename SpanBuf::off_type, typename TraitsT::off_type>);
  static_assert(std::is_same_v<typename SpanBuf::traits_type, TraitsT>);

  // Copy properties

  static_assert(!std::is_copy_constructible_v<SpanBuf>);
  static_assert(!std::is_copy_assignable_v<SpanBuf>);

  // Move properties

  static_assert(std::is_move_constructible_v<SpanBuf>);
  static_assert(std::is_move_assignable_v<SpanBuf>);
}

void test() {
#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_char, nasty_char_traits>();
#endif
  test<char, constexpr_char_traits<char>>();
  test<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, constexpr_char_traits<wchar_t>>();
  test<wchar_t, std::char_traits<wchar_t>>();
#endif
}

// Default template arguments

static_assert(std::is_same_v<std::basic_spanbuf<char>, std::basic_spanbuf<char, std::char_traits<char>>>);

// Aliases

static_assert(std::is_same_v<std::basic_spanbuf<char>, std::spanbuf>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_same_v<std::basic_spanbuf<wchar_t>, std::wspanbuf>);
#endif
