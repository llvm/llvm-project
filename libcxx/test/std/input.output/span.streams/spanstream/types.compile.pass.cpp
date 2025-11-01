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
//   class basic_spanstream
//     : public basic_iostream<charT, traits> {
//   public:
//     using char_type   = charT;
//     using int_type    = typename traits::int_type;
//     using pos_type    = typename traits::pos_type;
//     using off_type    = typename traits::off_type;
//     using traits_type = traits;

//   using spanstream = basic_spanstream<char>;
//   using wspanstream = basic_spanstream<wchar_t>;

#include <spanstream>
#include <string>
#include <type_traits>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT>
void test() {
  using SpStream = std::basic_spanstream<CharT, TraitsT>;

  // Constructors

  static_assert(!std::is_default_constructible_v<SpStream>);

  // Types

  static_assert(std::is_base_of_v<std::basic_iostream<CharT, TraitsT>, SpStream>);
  static_assert(std::is_same_v<typename SpStream::char_type, CharT>);
  static_assert(std::is_same_v<typename SpStream::int_type, typename TraitsT::int_type>);
  static_assert(std::is_same_v<typename SpStream::pos_type, typename TraitsT::pos_type>);
  static_assert(std::is_same_v<typename SpStream::off_type, typename TraitsT::off_type>);
  static_assert(std::is_same_v<typename SpStream::traits_type, TraitsT>);

  // Copy properties

  static_assert(!std::is_copy_constructible_v<SpStream>);
  static_assert(!std::is_copy_assignable_v<SpStream>);

  // Move properties

  static_assert(std::is_move_constructible_v<SpStream>);
  static_assert(std::is_move_assignable_v<SpStream>);
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

static_assert(std::is_same_v<std::basic_spanstream<char>, std::basic_spanstream<char, std::char_traits<char>>>);

// Aliases

static_assert(std::is_same_v<std::basic_spanstream<char>, std::spanstream>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_same_v<std::basic_spanstream<wchar_t>, std::wspanstream>);
#endif
