//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

#include <iterator>

#include <concepts>

static_assert(!std::__integer_like<bool>);
static_assert(std::__integer_like<signed char>);
static_assert(std::__integer_like<unsigned char>);
static_assert(std::__integer_like<short>);
static_assert(std::__integer_like<unsigned short>);
static_assert(std::__integer_like<int>);
static_assert(std::__integer_like<unsigned int>);
static_assert(std::__integer_like<long>);
static_assert(std::__integer_like<unsigned long>);
static_assert(std::__integer_like<long long>);
static_assert(std::__integer_like<unsigned long long>);
static_assert(std::__integer_like<char>);
static_assert(std::__integer_like<wchar_t>);
static_assert(std::__integer_like<char8_t>);
static_assert(std::__integer_like<char16_t>);
static_assert(std::__integer_like<char32_t>);

static_assert(!std::__signed_integer_like<bool>);
static_assert(std::__signed_integer_like<signed char>);
static_assert(std::__signed_integer_like<short>);
static_assert(std::__signed_integer_like<int>);
static_assert(std::__signed_integer_like<long>);
static_assert(std::__signed_integer_like<long long>);
static_assert(!std::__signed_integer_like<unsigned char>);
static_assert(!std::__signed_integer_like<unsigned short>);
static_assert(!std::__signed_integer_like<unsigned int>);
static_assert(!std::__signed_integer_like<unsigned long>);
static_assert(!std::__signed_integer_like<unsigned long long>);
static_assert(std::__signed_integer_like<char> == std::signed_integral<char>);
static_assert(std::__signed_integer_like<wchar_t> == std::signed_integral<wchar_t>);
static_assert(std::__signed_integer_like<char8_t> == std::signed_integral<char8_t>);
static_assert(std::__signed_integer_like<char16_t> == std::signed_integral<char16_t>);
static_assert(std::__signed_integer_like<char32_t> == std::signed_integral<char32_t>);
