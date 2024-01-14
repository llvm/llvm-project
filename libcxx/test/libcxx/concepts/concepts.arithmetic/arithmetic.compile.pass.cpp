//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <concepts>

struct SomeObject {};

// Concept helpers for the internal type traits for the fundamental types.

// template <class _Tp>
// concept __libcpp_unsigned_integer;

// Unsigned
static_assert(std::__libcpp_unsigned_integer<unsigned char>);
static_assert(std::__libcpp_unsigned_integer<unsigned short int>);
static_assert(std::__libcpp_unsigned_integer<unsigned int>);
static_assert(std::__libcpp_unsigned_integer<unsigned long int>);
static_assert(std::__libcpp_unsigned_integer<unsigned long long int>);
static_assert(std::__libcpp_unsigned_integer<unsigned short int>);
#ifndef _LIBCPP_HAS_NO_INT128
static_assert(std::__libcpp_unsigned_integer<__uint128_t>);
#endif
// Signed
static_assert(!std::__libcpp_unsigned_integer<signed char>);
static_assert(!std::__libcpp_unsigned_integer<short int>);
static_assert(!std::__libcpp_unsigned_integer<int>);
static_assert(!std::__libcpp_unsigned_integer<long int>);
static_assert(!std::__libcpp_unsigned_integer<long long int>);
static_assert(!std::__libcpp_unsigned_integer<short int>);
#ifndef _LIBCPP_HAS_NO_INT128
static_assert(!std::__libcpp_unsigned_integer<__int128_t>);
#endif
// Non-integer
static_assert(!std::__libcpp_unsigned_integer<bool>);
static_assert(!std::__libcpp_unsigned_integer<char8_t>);
static_assert(!std::__libcpp_unsigned_integer<char16_t>);
static_assert(!std::__libcpp_unsigned_integer<char32_t>);
static_assert(!std::__libcpp_unsigned_integer<float>);
static_assert(!std::__libcpp_unsigned_integer<double>);
static_assert(!std::__libcpp_unsigned_integer<long double>);
static_assert(!std::__libcpp_unsigned_integer<void>);
static_assert(!std::__libcpp_unsigned_integer<SomeObject>);

// template <class _Tp>
// concept __libcpp_signed_integer;

// Unsigned
static_assert(!std::__libcpp_signed_integer<unsigned char>);
static_assert(!std::__libcpp_signed_integer<unsigned short int>);
static_assert(!std::__libcpp_signed_integer<unsigned int>);
static_assert(!std::__libcpp_signed_integer<unsigned long int>);
static_assert(!std::__libcpp_signed_integer<unsigned long long int>);
static_assert(!std::__libcpp_signed_integer<unsigned short int>);
#ifndef _LIBCPP_HAS_NO_INT128
static_assert(!std::__libcpp_signed_integer<__uint128_t>);
#endif
// Signed
static_assert(std::__libcpp_signed_integer<signed char>);
static_assert(std::__libcpp_signed_integer<short int>);
static_assert(std::__libcpp_signed_integer<int>);
static_assert(std::__libcpp_signed_integer<long int>);
static_assert(std::__libcpp_signed_integer<long long int>);
static_assert(std::__libcpp_signed_integer<short int>);
#ifndef _LIBCPP_HAS_NO_INT128
static_assert(std::__libcpp_signed_integer<__int128_t>);
#endif
// Non-integer
static_assert(!std::__libcpp_signed_integer<bool>);
static_assert(!std::__libcpp_signed_integer<char8_t>);
static_assert(!std::__libcpp_signed_integer<char16_t>);
static_assert(!std::__libcpp_signed_integer<char32_t>);
static_assert(!std::__libcpp_signed_integer<float>);
static_assert(!std::__libcpp_signed_integer<double>);
static_assert(!std::__libcpp_signed_integer<long double>);
static_assert(!std::__libcpp_signed_integer<void>);
static_assert(!std::__libcpp_signed_integer<SomeObject>);

// template <class _Tp>
// concept __libcpp_integer;

// Unsigned
static_assert(std::__libcpp_integer<unsigned char>);
static_assert(std::__libcpp_integer<unsigned short int>);
static_assert(std::__libcpp_integer<unsigned int>);
static_assert(std::__libcpp_integer<unsigned long int>);
static_assert(std::__libcpp_integer<unsigned long long int>);
static_assert(std::__libcpp_integer<unsigned short int>);
#ifndef _LIBCPP_HAS_NO_INT128
static_assert(std::__libcpp_integer<__uint128_t>);
#endif
// Signed
static_assert(std::__libcpp_integer<signed char>);
static_assert(std::__libcpp_integer<short int>);
static_assert(std::__libcpp_integer<int>);
static_assert(std::__libcpp_integer<long int>);
static_assert(std::__libcpp_integer<long long int>);
static_assert(std::__libcpp_integer<short int>);
#ifndef _LIBCPP_HAS_NO_INT128
static_assert(std::__libcpp_integer<__int128_t>);
#endif
// Non-integer
static_assert(!std::__libcpp_integer<bool>);
static_assert(!std::__libcpp_integer<char8_t>);
static_assert(!std::__libcpp_integer<char16_t>);
static_assert(!std::__libcpp_integer<char32_t>);
static_assert(!std::__libcpp_integer<float>);
static_assert(!std::__libcpp_integer<double>);
static_assert(!std::__libcpp_integer<long double>);
static_assert(!std::__libcpp_integer<void>);
static_assert(!std::__libcpp_integer<SomeObject>);
