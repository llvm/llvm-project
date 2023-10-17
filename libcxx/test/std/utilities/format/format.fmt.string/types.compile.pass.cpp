//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// libc++ supports basic_format_string in C++20 as an extension
// UNSUPPORTED: !stdlib=libc++ && c++20

// <format>

//  template<class... Args>
//    using format_string =
//      basic_format_string<char, type_identity_t<Args>...>;
//  template<class... Args>
//    using wformat_string =
//      basic_format_string<wchar_t, type_identity_t<Args>...>;

#include <format>

#include <concepts>

#include "test_macros.h"

static_assert(std::same_as<std::format_string<>, std::basic_format_string<char>>);
static_assert(std::same_as<std::format_string<int>, std::basic_format_string<char, int>>);
static_assert(std::same_as<std::format_string<int, bool>, std::basic_format_string<char, int, bool>>);
static_assert(std::same_as<std::format_string<int, bool, void*>, std::basic_format_string<char, int, bool, void*>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::same_as<std::wformat_string<>, std::basic_format_string<wchar_t>>);
static_assert(std::same_as<std::wformat_string<int>, std::basic_format_string<wchar_t, int>>);
static_assert(std::same_as<std::wformat_string<int, bool>, std::basic_format_string<wchar_t, int, bool>>);
static_assert(std::same_as<std::wformat_string<int, bool, void*>, std::basic_format_string<wchar_t, int, bool, void*>>);
#endif
