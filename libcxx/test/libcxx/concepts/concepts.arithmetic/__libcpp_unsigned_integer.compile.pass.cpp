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
// concept __unsigned_integer;

#include <__type_traits/integer_traits.h>

#include "test_macros.h"

struct SomeObject {};

enum SomeEnum {};

enum class SomeScopedEnum {};

// Unsigned
static_assert(std::__unsigned_integer<unsigned char>);
static_assert(std::__unsigned_integer<unsigned short int>);
static_assert(std::__unsigned_integer<unsigned int>);
static_assert(std::__unsigned_integer<unsigned long int>);
static_assert(std::__unsigned_integer<unsigned long long int>);
static_assert(std::__unsigned_integer<unsigned short int>);
#if _LIBCPP_HAS_INT128
static_assert(std::__unsigned_integer<__uint128_t>);
#endif
// Signed
static_assert(!std::__unsigned_integer<signed char>);
static_assert(!std::__unsigned_integer<short int>);
static_assert(!std::__unsigned_integer<int>);
static_assert(!std::__unsigned_integer<long int>);
static_assert(!std::__unsigned_integer<long long int>);
static_assert(!std::__unsigned_integer<short int>);
#if _LIBCPP_HAS_INT128
static_assert(!std::__unsigned_integer<__int128_t>);
#endif
// Non-integer
static_assert(!std::__unsigned_integer<bool>);
static_assert(!std::__unsigned_integer<char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::__unsigned_integer<wchar_t>);
#endif
static_assert(!std::__unsigned_integer<char8_t>);
static_assert(!std::__unsigned_integer<char16_t>);
static_assert(!std::__unsigned_integer<char32_t>);
static_assert(!std::__unsigned_integer<float>);
static_assert(!std::__unsigned_integer<double>);
static_assert(!std::__unsigned_integer<long double>);
static_assert(!std::__unsigned_integer<void>);
static_assert(!std::__unsigned_integer<int*>);
static_assert(!std::__unsigned_integer<unsigned int*>);
static_assert(!std::__unsigned_integer<SomeObject>);
static_assert(!std::__unsigned_integer<SomeEnum>);
static_assert(!std::__unsigned_integer<SomeScopedEnum>);

// cv-qualified versions are distinct types ([basic.type.qualifier]) and so
// not unsigned integer types per [basic.fundamental]/p2. The three meaningful
// flavors in C++ are const, volatile, and const volatile.
static_assert(!std::__unsigned_integer<const unsigned int>);
static_assert(!std::__unsigned_integer<volatile unsigned int>);
static_assert(!std::__unsigned_integer<const volatile unsigned int>);
static_assert(!std::__unsigned_integer<const unsigned long long>);
static_assert(!std::__unsigned_integer<const unsigned char>);
static_assert(!std::__unsigned_integer<const char>);
static_assert(!std::__unsigned_integer<const bool>);
static_assert(!std::__unsigned_integer<const char8_t>);
static_assert(!std::__unsigned_integer<const char16_t>);
static_assert(!std::__unsigned_integer<const char32_t>);
static_assert(!std::__unsigned_integer<unsigned int&>);
static_assert(!std::__unsigned_integer<const unsigned int&>);

// Extended unsigned integer types per [basic.fundamental]/p3 Note 1.
#if TEST_HAS_EXTENSION(bit_int)
static_assert(std::__unsigned_integer<unsigned _BitInt(8)>);
static_assert(std::__unsigned_integer<unsigned _BitInt(16)>);
static_assert(std::__unsigned_integer<unsigned _BitInt(64)>);
static_assert(std::__unsigned_integer<unsigned _BitInt(13)>);
static_assert(!std::__unsigned_integer<signed _BitInt(16)>);
static_assert(!std::__unsigned_integer<const unsigned _BitInt(16)>);
static_assert(!std::__unsigned_integer<volatile unsigned _BitInt(64)>);
static_assert(!std::__unsigned_integer<const volatile unsigned _BitInt(13)>);
#  if __BITINT_MAXWIDTH__ >= 128
static_assert(std::__unsigned_integer<unsigned _BitInt(128)>);
static_assert(!std::__unsigned_integer<const unsigned _BitInt(128)>);
#  endif
#endif
