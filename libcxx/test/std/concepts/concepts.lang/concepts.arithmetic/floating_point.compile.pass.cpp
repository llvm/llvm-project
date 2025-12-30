//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// concept floating_point = // see below

#include <concepts>
#include <type_traits>

#include "arithmetic.h"

template <typename T>
constexpr bool CheckFloatingPointQualifiers() {
  constexpr bool result = std::floating_point<T>;
  static_assert(std::floating_point<const T> == result);
  static_assert(std::floating_point<volatile T> == result);
  static_assert(std::floating_point<const volatile T> == result);

  static_assert(!std::floating_point<T&>);
  static_assert(!std::floating_point<const T&>);
  static_assert(!std::floating_point<volatile T&>);
  static_assert(!std::floating_point<const volatile T&>);

  static_assert(!std::floating_point<T&&>);
  static_assert(!std::floating_point<const T&&>);
  static_assert(!std::floating_point<volatile T&&>);
  static_assert(!std::floating_point<const volatile T&&>);

  static_assert(!std::floating_point<T*>);
  static_assert(!std::floating_point<const T*>);
  static_assert(!std::floating_point<volatile T*>);
  static_assert(!std::floating_point<const volatile T*>);

  static_assert(!std::floating_point<T (*)()>);
  static_assert(!std::floating_point<T (&)()>);
  static_assert(!std::floating_point<T (&&)()>);

  return result;
}

// floating-point types
static_assert(CheckFloatingPointQualifiers<float>());
static_assert(CheckFloatingPointQualifiers<double>());
static_assert(CheckFloatingPointQualifiers<long double>());

// types that aren't floating-point
static_assert(!CheckFloatingPointQualifiers<signed char>());
static_assert(!CheckFloatingPointQualifiers<unsigned char>());
static_assert(!CheckFloatingPointQualifiers<short>());
static_assert(!CheckFloatingPointQualifiers<unsigned short>());
static_assert(!CheckFloatingPointQualifiers<int>());
static_assert(!CheckFloatingPointQualifiers<unsigned int>());
static_assert(!CheckFloatingPointQualifiers<long>());
static_assert(!CheckFloatingPointQualifiers<unsigned long>());
static_assert(!CheckFloatingPointQualifiers<long long>());
static_assert(!CheckFloatingPointQualifiers<unsigned long long>());
static_assert(!CheckFloatingPointQualifiers<wchar_t>());
static_assert(!CheckFloatingPointQualifiers<bool>());
static_assert(!CheckFloatingPointQualifiers<char>());
static_assert(!CheckFloatingPointQualifiers<char8_t>());
static_assert(!CheckFloatingPointQualifiers<char16_t>());
static_assert(!CheckFloatingPointQualifiers<char32_t>());
static_assert(!std::floating_point<void>);

static_assert(!CheckFloatingPointQualifiers<ClassicEnum>());
static_assert(!CheckFloatingPointQualifiers<ScopedEnum>());
static_assert(!CheckFloatingPointQualifiers<EmptyStruct>());
static_assert(!CheckFloatingPointQualifiers<int EmptyStruct::*>());
static_assert(!CheckFloatingPointQualifiers<int (EmptyStruct::*)()>());
