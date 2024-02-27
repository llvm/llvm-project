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

#include "test_macros.h"

// Types

static_assert(std::is_base_of_v<std::basic_streambuf<char>, std::spanbuf>);
static_assert(std::is_same_v<std::spanbuf::char_type, char>);
static_assert(std::is_same_v<std::spanbuf::int_type, std::char_traits<char>::int_type>);
static_assert(std::is_same_v<std::spanbuf::pos_type, std::char_traits<char>::pos_type>);
static_assert(std::is_same_v<std::spanbuf::off_type, std::char_traits<char>::off_type>);
static_assert(std::is_same_v<std::spanbuf::traits_type, std::char_traits<char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_base_of_v<std::basic_streambuf<wchar_t>, std::wspanbuf>);
static_assert(std::is_same_v<std::wspanbuf::char_type, wchar_t>);
static_assert(std::is_same_v<std::wspanbuf::int_type, std::char_traits<wchar_t>::int_type>);
static_assert(std::is_same_v<std::wspanbuf::pos_type, std::char_traits<wchar_t>::pos_type>);
static_assert(std::is_same_v<std::wspanbuf::off_type, std::char_traits<wchar_t>::off_type>);
static_assert(std::is_same_v<std::wspanbuf::traits_type, std::char_traits<wchar_t>>);
#endif

// Copy properties

static_assert(!std::is_copy_constructible_v<std::spanbuf>);
static_assert(!std::is_copy_assignable_v<std::spanbuf>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::is_copy_constructible_v<std::wspanbuf>);
static_assert(!std::is_copy_assignable_v<std::wspanbuf>);
#endif

// Move properties

static_assert(!std::is_copy_constructible_v<std::spanbuf>);
static_assert(!std::is_copy_assignable_v<std::spanbuf>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::is_copy_constructible_v<std::wspanbuf>);
static_assert(!std::is_copy_assignable_v<std::wspanbuf>);
#endif

// Move properties

static_assert(std::is_move_constructible_v<std::spanbuf>);
static_assert(std::is_move_assignable_v<std::spanbuf>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_move_constructible_v<std::wspanbuf>);
static_assert(std::is_move_assignable_v<std::wspanbuf>);
#endif

