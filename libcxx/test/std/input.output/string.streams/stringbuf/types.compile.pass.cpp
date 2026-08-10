//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf
//     : public basic_streambuf<charT, traits>
// {
// public:
//     typedef charT                          char_type;
//     typedef traits                         traits_type;
//     typedef typename traits_type::int_type int_type;
//     typedef typename traits_type::pos_type pos_type;
//     typedef typename traits_type::off_type off_type;
//     typedef Allocator                      allocator_type;
//
//     basic_stringbuf(const basic_stringbuf&) = delete;
//     basic_stringbuf& operator=(const basic_stringbuf&) = delete;
//
//     basic_stringbuf(basic_stringbuf&& rhs);
//     basic_stringbuf& operator=(basic_stringbuf&& rhs);

#include <sstream>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_base_of<std::basic_streambuf<char>, std::basic_stringbuf<char> >::value, "");
static_assert(std::is_same<std::basic_stringbuf<char>::char_type, char>::value, "");
static_assert(std::is_same<std::basic_stringbuf<char>::traits_type, std::char_traits<char> >::value, "");
static_assert(std::is_same<std::basic_stringbuf<char>::int_type, std::char_traits<char>::int_type>::value, "");
static_assert(std::is_same<std::basic_stringbuf<char>::pos_type, std::char_traits<char>::pos_type>::value, "");
static_assert(std::is_same<std::basic_stringbuf<char>::off_type, std::char_traits<char>::off_type>::value, "");
static_assert(std::is_same<std::basic_stringbuf<char>::allocator_type, std::allocator<char> >::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_base_of<std::basic_streambuf<wchar_t>, std::basic_stringbuf<wchar_t> >::value, "");
static_assert(std::is_same<std::basic_stringbuf<wchar_t>::char_type, wchar_t>::value, "");
static_assert(std::is_same<std::basic_stringbuf<wchar_t>::traits_type, std::char_traits<wchar_t> >::value, "");
static_assert(std::is_same<std::basic_stringbuf<wchar_t>::int_type, std::char_traits<wchar_t>::int_type>::value, "");
static_assert(std::is_same<std::basic_stringbuf<wchar_t>::pos_type, std::char_traits<wchar_t>::pos_type>::value, "");
static_assert(std::is_same<std::basic_stringbuf<wchar_t>::off_type, std::char_traits<wchar_t>::off_type>::value, "");
static_assert(std::is_same<std::basic_stringbuf<wchar_t>::allocator_type, std::allocator<wchar_t> >::value, "");
#endif

// Copy properties

static_assert(!std::is_copy_constructible<std::basic_stringbuf<char> >::value, "");
static_assert(!std::is_copy_assignable<std::basic_stringbuf<char> >::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::is_copy_constructible<std::basic_stringbuf<wchar_t> >::value, "");
static_assert(!std::is_copy_assignable<std::basic_stringbuf<wchar_t> >::value, "");
#endif

// Move properties

static_assert(std::is_move_constructible<std::basic_stringbuf<char> >::value, "");
static_assert(std::is_move_assignable<std::basic_stringbuf<char> >::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_move_constructible<std::basic_stringbuf<wchar_t> >::value, "");
static_assert(std::is_move_assignable<std::basic_stringbuf<wchar_t> >::value, "");
#endif
