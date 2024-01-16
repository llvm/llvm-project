//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <numeric>

// template<class R, class T>
//   constexpr R saturate_cast(T x) noexcept;                     // freestanding

#include <cassert>
#include <concepts>
#include <limits>
#include <numeric>

template <typename IntegerResultT, typename IntegerT>
constexpr bool test_signed_notsaturated() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::saturate_cast<IntegerResultT>(minVal)));
  static_assert(noexcept(std::saturate_cast<IntegerResultT>(maxVal)));

  assert(std::saturate_cast<IntegerResultT>(minVal) == minVal);
  assert(std::saturate_cast<IntegerResultT>(maxVal) == maxVal);

  return true;
}

template <typename IntegerResultT, typename IntegerT>
constexpr bool test_signed_saturated() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::saturate_cast<IntegerResultT>(minVal)));
  static_assert(noexcept(std::saturate_cast<IntegerResultT>(maxVal)));

  assert(std::saturate_cast<IntegerResultT>(minVal) == std::numeric_limits<IntegerResultT>::min());
  assert(std::saturate_cast<IntegerResultT>(maxVal) == std::numeric_limits<IntegerResultT>::max());

  return true;
}

template <typename IntegerResultT, typename IntegerT>
constexpr bool test_unsigned_notsaturated() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::saturate_cast<IntegerResultT>(minVal)));
  static_assert(noexcept(std::saturate_cast<IntegerResultT>(maxVal)));

  assert(std::saturate_cast<IntegerResultT>(minVal) == minVal);
  assert(std::saturate_cast<IntegerResultT>(maxVal) == maxVal);

  return true;
}

template <typename IntegerResultT, typename IntegerT>
constexpr bool test_unsigned_saturated() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::saturate_cast<IntegerResultT>(minVal)));
  static_assert(noexcept(std::saturate_cast<IntegerResultT>(maxVal)));

  assert(std::saturate_cast<IntegerResultT>(maxVal) == std::numeric_limits<IntegerResultT>::max());
  assert(std::saturate_cast<IntegerResultT>(maxVal) == std::numeric_limits<IntegerResultT>::max());

  return true;
}

constexpr bool test() {
  // Signed
  test_signed_notsaturated<long long int, signed char>();
  test_signed_notsaturated<long long int, short int>();
  test_signed_notsaturated<long long int, int>();
  test_signed_notsaturated<long long int, long int>();
  test_signed_notsaturated<long long int, long long int>();
  test_signed_saturated<signed char, long long int>();
  test_signed_saturated<short int, long long int>();
  test_signed_saturated<int, long long int>();
  test_signed_saturated<long int, long long int>();
  test_signed_saturated<long long int, long long int>();
#ifndef _LIBCPP_HAS_NO_INT128
  test_signed_notsaturated<__int128_t, signed char>();
  test_signed_notsaturated<__int128_t, short int>();
  test_signed_notsaturated<__int128_t, int>();
  test_signed_notsaturated<__int128_t, long int>();
  test_signed_notsaturated<__int128_t, long long int>();
  test_signed_saturated<signed char, __int128_t>();
  test_signed_saturated<short int, __int128_t>();
  test_signed_saturated<int, __int128_t>();
  test_signed_saturated<long int, __int128_t>();
  test_signed_saturated<long long int, __int128_t>();
#endif
  // Unsigned
  test_unsigned_notsaturated<unsigned long long int, unsigned char>();
  test_unsigned_notsaturated<unsigned long long int, unsigned short int>();
  test_unsigned_notsaturated<unsigned long long int, unsigned int>();
  test_unsigned_notsaturated<unsigned long long int, unsigned long int>();
  test_unsigned_notsaturated<unsigned long long int, unsigned long long int>();
  test_unsigned_saturated<unsigned char, unsigned long long int>();
  test_unsigned_saturated<unsigned short int, unsigned long long int>();
  test_unsigned_saturated<unsigned int, unsigned long long int>();
  test_unsigned_saturated<unsigned long int, unsigned long long int>();
  test_unsigned_saturated<unsigned long long int, unsigned long long int>();
#ifndef _LIBCPP_HAS_NO_INT128
  test_unsigned_notsaturated<__uint128_t, unsigned char>();
  test_unsigned_notsaturated<__uint128_t, unsigned short int>();
  test_unsigned_notsaturated<__uint128_t, unsigned int>();
  test_unsigned_notsaturated<__uint128_t, unsigned long int>();
  test_unsigned_notsaturated<__uint128_t, unsigned long long int>();
  test_unsigned_saturated<unsigned char, __uint128_t>();
  test_unsigned_saturated<unsigned short int, __uint128_t>();
  test_unsigned_saturated<unsigned int, __uint128_t>();
  test_unsigned_saturated<unsigned long int, __uint128_t>();
  test_unsigned_saturated<unsigned long long int, __uint128_t>();
#endif

  return true;
}

#include <print>

int main(int, char**) {
  assert(test());
  static_assert(test());

  // std::println(stderr, "-----> {}", std::saturate_cast<signed short>(999));
  // std::println(stderr, "-----> {}", std::saturate_cast<unsigned short>(999));
  // assert(false);

  return 0;
}

#include <numeric>
#include <climits>

#if CHAR_BIT == 8
static_assert(std::saturate_cast<unsigned char>(999) == 255);
static_assert(std::saturate_cast<signed char>(999) == 127);
#endif
static_assert(std::saturate_cast<unsigned short>(999) == 999);
static_assert(std::saturate_cast<signed short>(999) == 999);
static_assert(std::saturate_cast<short>(INT_MAX) == SHRT_MAX);
static_assert(std::saturate_cast<short>(UINT_MAX) == SHRT_MAX);
static_assert(std::saturate_cast<short>(UINT_MAX) == SHRT_MAX);
static_assert(std::saturate_cast<unsigned short>(UINT_MAX) == USHRT_MAX);
static_assert(std::saturate_cast<int>(UINT_MAX) == INT_MAX);
static_assert(std::saturate_cast<int>(INT_MAX) == INT_MAX);
static_assert(std::saturate_cast<unsigned>(-1) == 0);
static_assert(std::saturate_cast<unsigned>(INT_MIN) == 0);
static_assert(std::saturate_cast<unsigned>(UINT_MAX) == UINT_MAX);
static_assert(std::saturate_cast<unsigned>(LLONG_MAX) == UINT_MAX);
static_assert(std::saturate_cast<unsigned>(ULLONG_MAX) == UINT_MAX);
