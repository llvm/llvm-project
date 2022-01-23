//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept integral = // see below

#include <concepts>
#include <type_traits>

#include "arithmetic.h"
#include "test_macros.h"

template <typename T>
constexpr bool CheckIntegralQualifiers() {
  constexpr bool result = std::integral<T>;
  static_assert(std::integral<const T> == result);
  static_assert(std::integral<volatile T> == result);
  static_assert(std::integral<const volatile T> == result);

  static_assert(!std::integral<T&>);
  static_assert(!std::integral<const T&>);
  static_assert(!std::integral<volatile T&>);
  static_assert(!std::integral<const volatile T&>);

  static_assert(!std::integral<T&&>);
  static_assert(!std::integral<const T&&>);
  static_assert(!std::integral<volatile T&&>);
  static_assert(!std::integral<const volatile T&&>);

  static_assert(!std::integral<T*>);
  static_assert(!std::integral<const T*>);
  static_assert(!std::integral<volatile T*>);
  static_assert(!std::integral<const volatile T*>);

  static_assert(!std::integral<T (*)()>);
  static_assert(!std::integral<T (&)()>);
  static_assert(!std::integral<T(&&)()>);

  return result;
}

// standard signed and unsigned integers
static_assert(CheckIntegralQualifiers<signed char>());
static_assert(CheckIntegralQualifiers<unsigned char>());
static_assert(CheckIntegralQualifiers<short>());
static_assert(CheckIntegralQualifiers<unsigned short>());
static_assert(CheckIntegralQualifiers<int>());
static_assert(CheckIntegralQualifiers<unsigned int>());
static_assert(CheckIntegralQualifiers<long>());
static_assert(CheckIntegralQualifiers<unsigned long>());
static_assert(CheckIntegralQualifiers<long long>());
static_assert(CheckIntegralQualifiers<unsigned long long>());

// extended integers
#ifndef TEST_HAS_NO_INT128
static_assert(CheckIntegralQualifiers<__int128_t>());
static_assert(CheckIntegralQualifiers<__uint128_t>());
#endif

// bool and char types are also integral
static_assert(CheckIntegralQualifiers<wchar_t>());
static_assert(CheckIntegralQualifiers<bool>());
static_assert(CheckIntegralQualifiers<char>());
static_assert(CheckIntegralQualifiers<char8_t>());
static_assert(CheckIntegralQualifiers<char16_t>());
static_assert(CheckIntegralQualifiers<char32_t>());

// types that aren't integral
static_assert(!std::integral<void>);
static_assert(!CheckIntegralQualifiers<float>());
static_assert(!CheckIntegralQualifiers<double>());
static_assert(!CheckIntegralQualifiers<long double>());

static_assert(!CheckIntegralQualifiers<ClassicEnum>());

static_assert(!CheckIntegralQualifiers<ScopedEnum>());

static_assert(!CheckIntegralQualifiers<EmptyStruct>());
static_assert(!CheckIntegralQualifiers<int EmptyStruct::*>());
static_assert(!CheckIntegralQualifiers<int (EmptyStruct::*)()>());

static_assert(CheckSubsumption(0));
static_assert(CheckSubsumption(0U));

int main(int, char**) { return 0; }
