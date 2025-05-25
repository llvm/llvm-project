//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// concept unsigned_integral = // see below

#include <concepts>
#include <type_traits>

#include "arithmetic.h"
#include "test_macros.h"

template <typename T>
constexpr bool CheckUnsignedIntegralQualifiers() {
  constexpr bool result = std::unsigned_integral<T>;
  static_assert(std::unsigned_integral<const T> == result);
  static_assert(std::unsigned_integral<volatile T> == result);
  static_assert(std::unsigned_integral<const volatile T> == result);

  static_assert(!std::unsigned_integral<T&>);
  static_assert(!std::unsigned_integral<const T&>);
  static_assert(!std::unsigned_integral<volatile T&>);
  static_assert(!std::unsigned_integral<const volatile T&>);

  static_assert(!std::unsigned_integral<T&&>);
  static_assert(!std::unsigned_integral<const T&&>);
  static_assert(!std::unsigned_integral<volatile T&&>);
  static_assert(!std::unsigned_integral<const volatile T&&>);

  static_assert(!std::unsigned_integral<T*>);
  static_assert(!std::unsigned_integral<const T*>);
  static_assert(!std::unsigned_integral<volatile T*>);
  static_assert(!std::unsigned_integral<const volatile T*>);

  static_assert(!std::unsigned_integral<T (*)()>);
  static_assert(!std::unsigned_integral<T (&)()>);
  static_assert(!std::unsigned_integral<T (&&)()>);

  return result;
}

// standard unsigned types
static_assert(CheckUnsignedIntegralQualifiers<unsigned char>());
static_assert(CheckUnsignedIntegralQualifiers<unsigned short>());
static_assert(CheckUnsignedIntegralQualifiers<unsigned int>());
static_assert(CheckUnsignedIntegralQualifiers<unsigned long>());
static_assert(CheckUnsignedIntegralQualifiers<unsigned long long>());

// Whether bool and character types are signed or unsigned is impl-defined
static_assert(CheckUnsignedIntegralQualifiers<wchar_t>() == !std::is_signed_v<wchar_t>);
static_assert(CheckUnsignedIntegralQualifiers<bool>() == !std::is_signed_v<bool>);
static_assert(CheckUnsignedIntegralQualifiers<char>() == !std::is_signed_v<char>);
static_assert(CheckUnsignedIntegralQualifiers<char8_t>() == !std::is_signed_v<char8_t>);
static_assert(CheckUnsignedIntegralQualifiers<char16_t>() == !std::is_signed_v<char16_t>);
static_assert(CheckUnsignedIntegralQualifiers<char32_t>() == !std::is_signed_v<char32_t>);

// extended integers
#ifndef TEST_HAS_NO_INT128
static_assert(CheckUnsignedIntegralQualifiers<__uint128_t>());
static_assert(!CheckUnsignedIntegralQualifiers<__int128_t>());
#endif

// integer types that aren't unsigned integrals
static_assert(!CheckUnsignedIntegralQualifiers<signed char>());
static_assert(!CheckUnsignedIntegralQualifiers<short>());
static_assert(!CheckUnsignedIntegralQualifiers<int>());
static_assert(!CheckUnsignedIntegralQualifiers<long>());
static_assert(!CheckUnsignedIntegralQualifiers<long long>());

static_assert(!std::unsigned_integral<void>);
static_assert(!CheckUnsignedIntegralQualifiers<float>());
static_assert(!CheckUnsignedIntegralQualifiers<double>());
static_assert(!CheckUnsignedIntegralQualifiers<long double>());

static_assert(!CheckUnsignedIntegralQualifiers<ClassicEnum>());
static_assert(!CheckUnsignedIntegralQualifiers<ScopedEnum>());
static_assert(!CheckUnsignedIntegralQualifiers<EmptyStruct>());
static_assert(!CheckUnsignedIntegralQualifiers<int EmptyStruct::*>());
static_assert(!CheckUnsignedIntegralQualifiers<int (EmptyStruct::*)()>());

static_assert(CheckSubsumption(0));
static_assert(CheckSubsumption(0U));
