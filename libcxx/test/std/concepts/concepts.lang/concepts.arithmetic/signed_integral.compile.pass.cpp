//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// concept signed_integral = // see below

#include <concepts>
#include <type_traits>

#include "arithmetic.h"
#include "test_macros.h"

template <typename T>
constexpr bool CheckSignedIntegralQualifiers() {
  constexpr bool result = std::signed_integral<T>;
  static_assert(std::signed_integral<const T> == result);
  static_assert(std::signed_integral<volatile T> == result);
  static_assert(std::signed_integral<const volatile T> == result);

  static_assert(!std::signed_integral<T&>);
  static_assert(!std::signed_integral<const T&>);
  static_assert(!std::signed_integral<volatile T&>);
  static_assert(!std::signed_integral<const volatile T&>);

  static_assert(!std::signed_integral<T&&>);
  static_assert(!std::signed_integral<const T&&>);
  static_assert(!std::signed_integral<volatile T&&>);
  static_assert(!std::signed_integral<const volatile T&&>);

  static_assert(!std::signed_integral<T*>);
  static_assert(!std::signed_integral<const T*>);
  static_assert(!std::signed_integral<volatile T*>);
  static_assert(!std::signed_integral<const volatile T*>);

  static_assert(!std::signed_integral<T (*)()>);
  static_assert(!std::signed_integral<T (&)()>);
  static_assert(!std::signed_integral<T (&&)()>);

  return result;
}

// standard signed integers
static_assert(CheckSignedIntegralQualifiers<signed char>());
static_assert(CheckSignedIntegralQualifiers<short>());
static_assert(CheckSignedIntegralQualifiers<int>());
static_assert(CheckSignedIntegralQualifiers<long>());
static_assert(CheckSignedIntegralQualifiers<long long>());

// bool and character *may* be signed
static_assert(CheckSignedIntegralQualifiers<wchar_t>() == std::is_signed_v<wchar_t>);
static_assert(CheckSignedIntegralQualifiers<bool>() == std::is_signed_v<bool>);
static_assert(CheckSignedIntegralQualifiers<char>() == std::is_signed_v<char>);
static_assert(CheckSignedIntegralQualifiers<char8_t>() == std::is_signed_v<char8_t>);
static_assert(CheckSignedIntegralQualifiers<char16_t>() == std::is_signed_v<char16_t>);
static_assert(CheckSignedIntegralQualifiers<char32_t>() == std::is_signed_v<char32_t>);

// integers that aren't signed integrals
static_assert(!CheckSignedIntegralQualifiers<unsigned char>());
static_assert(!CheckSignedIntegralQualifiers<unsigned short>());
static_assert(!CheckSignedIntegralQualifiers<unsigned int>());
static_assert(!CheckSignedIntegralQualifiers<unsigned long>());
static_assert(!CheckSignedIntegralQualifiers<unsigned long long>());

// extended integers
#ifndef TEST_HAS_NO_INT128
static_assert(CheckSignedIntegralQualifiers<__int128_t>());
static_assert(!CheckSignedIntegralQualifiers<__uint128_t>());
#endif

// types that aren't even integers shouldn't be signed integers!
static_assert(!std::signed_integral<void>);
static_assert(!CheckSignedIntegralQualifiers<float>());
static_assert(!CheckSignedIntegralQualifiers<double>());
static_assert(!CheckSignedIntegralQualifiers<long double>());

static_assert(!CheckSignedIntegralQualifiers<ClassicEnum>());
static_assert(!CheckSignedIntegralQualifiers<ScopedEnum>());
static_assert(!CheckSignedIntegralQualifiers<EmptyStruct>());
static_assert(!CheckSignedIntegralQualifiers<int EmptyStruct::*>());
static_assert(!CheckSignedIntegralQualifiers<int (EmptyStruct::*)()>());

static_assert(CheckSubsumption(0));
static_assert(CheckSubsumption(0U));
