//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Concept helpers for the internal type traits for the fundamental types.

// template <class _Tp>
// concept __signed_integer;

#include <__type_traits/integer_traits.h>

#include "test_macros.h"

struct SomeObject {};

enum SomeEnum {};

enum class SomeScopedEnum {};

// Unsigned
static_assert(!std::__signed_integer<unsigned char>);
static_assert(!std::__signed_integer<unsigned short int>);
static_assert(!std::__signed_integer<unsigned int>);
static_assert(!std::__signed_integer<unsigned long int>);
static_assert(!std::__signed_integer<unsigned long long int>);
static_assert(!std::__signed_integer<unsigned short int>);
#if _LIBCPP_HAS_INT128
static_assert(!std::__signed_integer<__uint128_t>);
#endif
// Signed
static_assert(std::__signed_integer<signed char>);
static_assert(std::__signed_integer<short int>);
static_assert(std::__signed_integer<int>);
static_assert(std::__signed_integer<long int>);
static_assert(std::__signed_integer<long long int>);
static_assert(std::__signed_integer<short int>);
#if _LIBCPP_HAS_INT128
static_assert(std::__signed_integer<__int128_t>);
#endif
// Non-integer
static_assert(!std::__signed_integer<bool>);
static_assert(!std::__signed_integer<char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::__signed_integer<wchar_t>);
#endif
static_assert(!std::__signed_integer<char8_t>);
static_assert(!std::__signed_integer<char16_t>);
static_assert(!std::__signed_integer<char32_t>);
static_assert(!std::__signed_integer<float>);
static_assert(!std::__signed_integer<double>);
static_assert(!std::__signed_integer<long double>);
static_assert(!std::__signed_integer<void>);
static_assert(!std::__signed_integer<int*>);
static_assert(!std::__signed_integer<unsigned int*>);
static_assert(!std::__signed_integer<SomeObject>);
static_assert(!std::__signed_integer<SomeEnum>);
static_assert(!std::__signed_integer<SomeScopedEnum>);
