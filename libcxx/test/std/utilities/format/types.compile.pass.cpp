//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// C++20
// ...  provides the following enabled specializations:
//  The debug-enabled specializations
//    template<> struct formatter<char, char>;
//    template<> struct formatter<char, wchar_t>;
//    template<> struct formatter<wchar_t, wchar_t>;
//
//  For each charT, the debug-enabled string type specializations template<>
//  struct formatter<charT*, charT>;
//    template<> struct formatter<const charT*, charT>;
//    template<size_t N> struct formatter<charT[N], charT>;
//    template<class traits, class Allocator>
//      struct formatter<basic_string<charT, traits, Allocator>, charT>;
//    template<class traits>
//      struct formatter<basic_string_view<charT, traits>, charT>;
//
//  For each charT, for each cv-unqualified arithmetic type ArithmeticT other
//  than char, wchar_t, char8_t, char16_t, or char32_t, a specialization
//    template<> struct formatter<ArithmeticT, charT>;
//
//  For each charT, the pointer type specializations template<> struct
//  formatter<nullptr_t, charT>;
//    template<> struct formatter<void*, charT>;
//    template<> struct formatter<const void*, charT>;

// C++23
// [format.range.formatter]
//   template<class T, class charT = char>
//     requires same_as<remove_cvref_t<T>, T> && formattable<T, charT>
//   class range_formatter;
//
// [format.tuple]/1
//   For each of pair and tuple, the library provides the following formatter
//   specialization where pair-or-tuple is the name of the template:
//   template<class charT, formattable<charT>... Ts>
//   struct formatter<pair-or-tuple<Ts...>, charT> {

// [format.formatter.spec]/4
//   If the library provides an explicit or partial specialization of
//   formatter<T, charT>, that specialization is enabled and meets the
//   Formatter requirements except as noted otherwise.
//
// Test parts of the BasicFormatter requirements. Like the formattable concept
// it uses the semiregular concept. This test does not use the formattable
// concept since the intent is for the formatter to be available without
// including the <format> header.

#include <concepts>
#include <cstddef>
#include <format>
#include <string_view>
#include <string>
#include <tuple>
#include <utility>

#include "test_macros.h"

static_assert(std::semiregular<std::formatter<char, char>>);

static_assert(std::semiregular<std::formatter<char*, char>>);
static_assert(std::semiregular<std::formatter<const char*, char>>);
static_assert(std::semiregular<std::formatter<char[1], char>>);
static_assert(std::semiregular<std::formatter<std::string, char>>);
static_assert(std::semiregular<std::formatter<std::string_view, char>>);

static_assert(std::semiregular<std::formatter<bool, char>>);

static_assert(std::semiregular<std::formatter<signed char, char>>);
static_assert(std::semiregular<std::formatter<signed short, char>>);
static_assert(std::semiregular<std::formatter<signed int, char>>);
static_assert(std::semiregular<std::formatter<signed long, char>>);
static_assert(std::semiregular<std::formatter<signed long long, char>>);

static_assert(std::semiregular<std::formatter<unsigned char, char>>);
static_assert(std::semiregular<std::formatter<unsigned short, char>>);
static_assert(std::semiregular<std::formatter<unsigned int, char>>);
static_assert(std::semiregular<std::formatter<unsigned long, char>>);
static_assert(std::semiregular<std::formatter<unsigned long long, char>>);

static_assert(std::semiregular<std::formatter<float, char>>);
static_assert(std::semiregular<std::formatter<double, char>>);
static_assert(std::semiregular<std::formatter<long double, char>>);

static_assert(std::semiregular<std::formatter<std::nullptr_t, char>>);
static_assert(std::semiregular<std::formatter<void*, char>>);
static_assert(std::semiregular<std::formatter<const void*, char>>);

#if TEST_STD_VER > 20
static_assert(std::semiregular<std::range_formatter<int, char>>);
static_assert(std::semiregular<std::formatter<std::tuple<int>, char>>);
static_assert(std::semiregular<std::range_formatter<std::pair<int, int>, char>>);
#endif

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::semiregular<std::formatter<char, wchar_t>>);
static_assert(std::semiregular<std::formatter<wchar_t, wchar_t>>);

static_assert(std::semiregular<std::formatter<wchar_t*, wchar_t>>);
static_assert(std::semiregular<std::formatter<const wchar_t*, wchar_t>>);
static_assert(std::semiregular<std::formatter<wchar_t[1], wchar_t>>);
static_assert(std::semiregular<std::formatter<std::wstring, wchar_t>>);
static_assert(std::semiregular<std::formatter<std::wstring_view, wchar_t>>);

static_assert(std::semiregular<std::formatter<bool, wchar_t>>);

static_assert(std::semiregular<std::formatter<signed char, wchar_t>>);
static_assert(std::semiregular<std::formatter<signed short, wchar_t>>);
static_assert(std::semiregular<std::formatter<signed int, wchar_t>>);
static_assert(std::semiregular<std::formatter<signed long, wchar_t>>);
static_assert(std::semiregular<std::formatter<signed long long, wchar_t>>);

static_assert(std::semiregular<std::formatter<unsigned char, wchar_t>>);
static_assert(std::semiregular<std::formatter<unsigned short, wchar_t>>);
static_assert(std::semiregular<std::formatter<unsigned int, wchar_t>>);
static_assert(std::semiregular<std::formatter<unsigned long, wchar_t>>);
static_assert(std::semiregular<std::formatter<unsigned long long, wchar_t>>);

static_assert(std::semiregular<std::formatter<float, wchar_t>>);
static_assert(std::semiregular<std::formatter<double, wchar_t>>);
static_assert(std::semiregular<std::formatter<long double, wchar_t>>);

static_assert(std::semiregular<std::formatter<std::nullptr_t, wchar_t>>);
static_assert(std::semiregular<std::formatter<void*, wchar_t>>);
static_assert(std::semiregular<std::formatter<const void*, wchar_t>>);

#  if TEST_STD_VER > 20
static_assert(std::semiregular<std::range_formatter<int, wchar_t>>);
static_assert(std::semiregular<std::formatter<std::tuple<int>, wchar_t>>);
static_assert(std::semiregular<std::range_formatter<std::pair<int, int>, wchar_t>>);
#  endif
#endif // TEST_HAS_NO_WIDE_CHARACTERS
