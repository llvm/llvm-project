//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// <syncstream>
//
//  template<class charT, class traits = char_traits<charT>, class Allocator = allocator<charT>>
//  class basic_osyncstream : public basic_ostream<charT, traits> {
//  public:
//    using char_type   = charT;
//    using int_type    = typename traits::int_type;
//    using pos_type    = typename traits::pos_type;
//    using off_type    = typename traits::off_type;
//    using traits_type = traits;
//
//    using allocator_type = Allocator;
//    using streambuf_type = basic_streambuf<charT, traits>;
//    using syncbuf_type   = basic_syncbuf<charT, traits, Allocator>;

#include <syncstream>
#include <concepts>

#include "test_macros.h"
#include "constexpr_char_traits.h"
#include "test_allocator.h"

static_assert(std::same_as<std::basic_osyncstream<char>,
                           std::basic_osyncstream<char, std::char_traits<char>, std::allocator<char>>>);
static_assert(std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>>,
                           std::basic_osyncstream<char, constexpr_char_traits<char>, std::allocator<char>>>);

static_assert(
    std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>, test_allocator<char>>::char_type, char>);
static_assert(std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>, test_allocator<char>>::int_type,
                           constexpr_char_traits<char>::int_type>);
static_assert(std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>, test_allocator<char>>::pos_type,
                           constexpr_char_traits<char>::pos_type>);
static_assert(std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>, test_allocator<char>>::off_type,
                           constexpr_char_traits<char>::off_type>);
static_assert(std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>, test_allocator<char>>::traits_type,
                           constexpr_char_traits<char>>);
static_assert(
    std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>, test_allocator<char>>::allocator_type,
                 test_allocator<char>>);
static_assert(
    std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>, test_allocator<char>>::streambuf_type,
                 std::basic_streambuf<char, constexpr_char_traits<char>>>);
static_assert(
    std::same_as<std::basic_osyncstream<char, constexpr_char_traits<char>, test_allocator<char>>::syncbuf_type,
                 std::basic_syncbuf<char, constexpr_char_traits<char>, test_allocator<char>>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS

static_assert(std::same_as<std::basic_osyncstream<wchar_t>,
                           std::basic_osyncstream<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>>>);
static_assert(std::same_as<std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>>,
                           std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, std::allocator<wchar_t>>>);

static_assert(
    std::same_as<std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>::char_type,
                 wchar_t>);
static_assert(
    std::same_as<std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>::int_type,
                 constexpr_char_traits<wchar_t>::int_type>);
static_assert(
    std::same_as<std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>::pos_type,
                 constexpr_char_traits<wchar_t>::pos_type>);
static_assert(
    std::same_as<std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>::off_type,
                 constexpr_char_traits<wchar_t>::off_type>);
static_assert(
    std::same_as<std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>::traits_type,
                 constexpr_char_traits<wchar_t>>);
static_assert(std::same_as<
              std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>::allocator_type,
              test_allocator<wchar_t>>);
static_assert(std::same_as<
              std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>::streambuf_type,
              std::basic_streambuf<wchar_t, constexpr_char_traits<wchar_t>>>);
static_assert(
    std::same_as<std::basic_osyncstream<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>::syncbuf_type,
                 std::basic_syncbuf<wchar_t, constexpr_char_traits<wchar_t>, test_allocator<wchar_t>>>);

#endif // TEST_HAS_NO_WIDE_CHARACTERS
