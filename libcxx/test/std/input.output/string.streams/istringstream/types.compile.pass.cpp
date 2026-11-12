//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_istringstream
//     : public basic_istream<charT, traits>
// {
// public:
//     typedef charT                          char_type;
//     typedef traits                         traits_type;
//     typedef typename traits_type::int_type int_type;
//     typedef typename traits_type::pos_type pos_type;
//     typedef typename traits_type::off_type off_type;
//     typedef Allocator                      allocator_type;
//
//     basic_istringstream(const basic_istringstream&) = delete;
//     basic_istringstream& operator=(const basic_istringstream&) = delete;
//
//     basic_istringstream(basic_istringstream&& rhs);
//     basic_istringstream& operator=(basic_istringstream&& rhs);

#include <sstream>
#include <type_traits>

#include "test_macros.h"

// Types

static_assert(std::is_base_of<std::basic_istream<char>, std::basic_istringstream<char> >::value, "");
static_assert(std::is_same<std::basic_istringstream<char>::char_type, char>::value, "");
static_assert(std::is_same<std::basic_istringstream<char>::traits_type, std::char_traits<char> >::value, "");
static_assert(std::is_same<std::basic_istringstream<char>::int_type, std::char_traits<char>::int_type>::value, "");
static_assert(std::is_same<std::basic_istringstream<char>::pos_type, std::char_traits<char>::pos_type>::value, "");
static_assert(std::is_same<std::basic_istringstream<char>::off_type, std::char_traits<char>::off_type>::value, "");
static_assert(std::is_same<std::basic_istringstream<char>::allocator_type, std::allocator<char> >::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_base_of<std::basic_istream<wchar_t>, std::basic_istringstream<wchar_t> >::value, "");
static_assert(std::is_same<std::basic_istringstream<wchar_t>::char_type, wchar_t>::value, "");
static_assert(std::is_same<std::basic_istringstream<wchar_t>::traits_type, std::char_traits<wchar_t> >::value, "");
static_assert(std::is_same<std::basic_istringstream<wchar_t>::int_type, std::char_traits<wchar_t>::int_type>::value,
              "");
static_assert(std::is_same<std::basic_istringstream<wchar_t>::pos_type, std::char_traits<wchar_t>::pos_type>::value,
              "");
static_assert(std::is_same<std::basic_istringstream<wchar_t>::off_type, std::char_traits<wchar_t>::off_type>::value,
              "");
static_assert(std::is_same<std::basic_istringstream<wchar_t>::allocator_type, std::allocator<wchar_t> >::value, "");
#endif

// Copy properties

static_assert(!std::is_copy_constructible<std::istringstream>::value, "");
static_assert(!std::is_copy_assignable<std::istringstream>::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::is_copy_constructible<std::wistringstream>::value, "");
static_assert(!std::is_copy_assignable<std::wistringstream>::value, "");
#endif

// Move properties

static_assert(std::is_move_constructible<std::istringstream>::value, "");
static_assert(std::is_move_assignable<std::istringstream>::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_move_constructible<std::wistringstream>::value, "");
static_assert(std::is_move_assignable<std::wistringstream>::value, "");
#endif
