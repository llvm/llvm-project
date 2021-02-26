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

// template<class T>
// concept signed_integral = // see below

// template<class T>
// concept unsigned_integral = // see below

// template<class T>
// concept floating_point = // see below

#include <concepts>
#include <type_traits>

namespace {
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

enum ClassicEnum { a, b, c };
enum class ScopedEnum { x, y, z };
struct EmptyStruct {};

constexpr void CheckIntegral() {
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
#ifndef _LIBCPP_HAS_NO_INT128
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
}

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
  static_assert(!std::signed_integral<T(&&)()>);

  return result;
}

constexpr void CheckSignedIntegral() {
  // standard signed integers
  static_assert(CheckSignedIntegralQualifiers<signed char>());
  static_assert(CheckSignedIntegralQualifiers<short>());
  static_assert(CheckSignedIntegralQualifiers<int>());
  static_assert(CheckSignedIntegralQualifiers<long>());
  static_assert(CheckSignedIntegralQualifiers<long long>());

  // bool and character *may* be signed
  static_assert(CheckSignedIntegralQualifiers<wchar_t>() ==
                std::is_signed_v<wchar_t>);
  static_assert(CheckSignedIntegralQualifiers<bool>() ==
                std::is_signed_v<bool>);
  static_assert(CheckSignedIntegralQualifiers<char>() ==
                std::is_signed_v<char>);
  static_assert(CheckSignedIntegralQualifiers<char8_t>() ==
                std::is_signed_v<char8_t>);
  static_assert(CheckSignedIntegralQualifiers<char16_t>() ==
                std::is_signed_v<char16_t>);
  static_assert(CheckSignedIntegralQualifiers<char32_t>() ==
                std::is_signed_v<char32_t>);

  // integers that aren't signed integrals
  static_assert(!CheckSignedIntegralQualifiers<unsigned char>());
  static_assert(!CheckSignedIntegralQualifiers<unsigned short>());
  static_assert(!CheckSignedIntegralQualifiers<unsigned int>());
  static_assert(!CheckSignedIntegralQualifiers<unsigned long>());
  static_assert(!CheckSignedIntegralQualifiers<unsigned long long>());

  // extended integers
#ifndef _LIBCPP_HAS_NO_INT128
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
}

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
  static_assert(!std::unsigned_integral<T(&&)()>);

  return result;
}

constexpr void CheckUnsignedIntegral() {
  // standard unsigned types
  static_assert(CheckUnsignedIntegralQualifiers<unsigned char>());
  static_assert(CheckUnsignedIntegralQualifiers<unsigned short>());
  static_assert(CheckUnsignedIntegralQualifiers<unsigned int>());
  static_assert(CheckUnsignedIntegralQualifiers<unsigned long>());
  static_assert(CheckUnsignedIntegralQualifiers<unsigned long long>());

  // Whether bool and character types are signed or unsigned is impl-defined
  static_assert(CheckUnsignedIntegralQualifiers<wchar_t>() ==
                !std::is_signed_v<wchar_t>);
  static_assert(CheckUnsignedIntegralQualifiers<bool>() ==
                !std::is_signed_v<bool>);
  static_assert(CheckUnsignedIntegralQualifiers<char>() ==
                !std::is_signed_v<char>);
  static_assert(CheckUnsignedIntegralQualifiers<char8_t>() ==
                !std::is_signed_v<char8_t>);
  static_assert(CheckUnsignedIntegralQualifiers<char16_t>() ==
                !std::is_signed_v<char16_t>);
  static_assert(CheckUnsignedIntegralQualifiers<char32_t>() ==
                !std::is_signed_v<char32_t>);

  // extended integers
#ifndef _LIBCPP_HAS_NO_INT128
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
}

// This overload should never be called. It exists solely to force subsumption.
template <std::integral I>
[[nodiscard]] constexpr bool CheckSubsumption(I) {
  return false;
}

// clang-format off
template <std::integral I>
requires std::signed_integral<I> && (!std::unsigned_integral<I>)
[[nodiscard]] constexpr bool CheckSubsumption(I) {
  return std::is_signed_v<I>;
}

template <std::integral I>
requires std::unsigned_integral<I> && (!std::signed_integral<I>)
[[nodiscard]] constexpr bool CheckSubsumption(I) {
  return std::is_unsigned_v<I>;
}
// clang-format on

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
  static_assert(!std::floating_point<T(&&)()>);

  return result;
}

constexpr void CheckFloatingPoint() {
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
}
} // namespace

int main(int, char**) {
  CheckIntegral();
  CheckSignedIntegral();
  CheckUnsignedIntegral();
  static_assert(CheckSubsumption(0));
  static_assert(CheckSubsumption(0U));
  CheckFloatingPoint();
  return 0;
}
