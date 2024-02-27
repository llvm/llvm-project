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

#include "test_macros.h"

// Types

static_assert(std::is_base_of_v<std::basic_iostream<char>, std::spanstream>);
static_assert(std::is_same_v<std::spanstream::char_type, char>);
static_assert(std::is_same_v<std::spanstream::int_type, std::char_traits<char>::int_type>);
static_assert(std::is_same_v<std::spanstream::pos_type, std::char_traits<char>::pos_type>);
static_assert(std::is_same_v<std::spanstream::off_type, std::char_traits<char>::off_type>);
static_assert(std::is_same_v<std::spanstream::traits_type, std::char_traits<char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_base_of_v<std::basic_iostream<wchar_t>, std::wspanstream>);
static_assert(std::is_same_v<std::wspanstream::char_type, wchar_t>);
static_assert(std::is_same_v<std::wspanstream::int_type, std::char_traits<wchar_t>::int_type>);
static_assert(std::is_same_v<std::wspanstream::pos_type, std::char_traits<wchar_t>::pos_type>);
static_assert(std::is_same_v<std::wspanstream::off_type, std::char_traits<wchar_t>::off_type>);
static_assert(std::is_same_v<std::wspanstream::traits_type, std::char_traits<wchar_t>>);
#endif

// Copy properties

static_assert(!std::is_copy_constructible_v<std::spanstream>);
static_assert(!std::is_copy_assignable_v<std::spanstream>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::is_copy_constructible_v<std::wspanstream>);
static_assert(!std::is_copy_assignable_v<std::wspanstream>);
#endif

// Move properties

static_assert(!std::is_copy_constructible_v<std::spanstream>);
static_assert(!std::is_copy_assignable_v<std::spanstream>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::is_copy_constructible_v<std::wspanstream>);
static_assert(!std::is_copy_assignable_v<std::wspanstream>);
#endif

// Move properties

static_assert(std::is_move_constructible_v<std::spanstream>);
static_assert(std::is_move_assignable_v<std::spanstream>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_move_constructible_v<std::wspanstream>);
static_assert(std::is_move_assignable_v<std::wspanstream>);
#endif
