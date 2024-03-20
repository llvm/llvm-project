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
#include <climits>
#include <concepts>
#include <limits>
#include <numeric>

#include "test_macros.h"
#include <print>

// Smaller to larger
static_assert(noexcept(std::saturate_cast<signed int>(std::numeric_limits<signed char>::max())));
static_assert(noexcept(std::saturate_cast<signed int>(std::numeric_limits<unsigned char>::max())));

static_assert(noexcept(std::saturate_cast<unsigned int>(std::numeric_limits<signed char>::max())));
static_assert(noexcept(std::saturate_cast<unsigned int>(std::numeric_limits<unsigned char>::max())));

// Same type
static_assert(noexcept(std::saturate_cast<signed long int>(std::numeric_limits<signed long int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned long int>(std::numeric_limits<unsigned long int>::max())));

// Larger to smaller
static_assert(noexcept(std::saturate_cast<signed char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(std::saturate_cast<signed char>(std::numeric_limits<unsigned int>::max())));

static_assert(noexcept(std::saturate_cast<unsigned char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned char>(std::numeric_limits<unsigned int>::max())));

// Tests

constexpr bool test() {
  // clang-format off

#ifndef TEST_HAS_NO_INT128
  using SIntT = __int128_t;
  using UIntT = __uint128_t;
#else
  using SIntT = long long int;
  using UIntT = unsigned long long int;
#endif

  // Constants the values of which depend on the context (platform)

  constexpr auto sBigMin = std::numeric_limits<SIntT>::min();
  constexpr auto sZero   = SIntT{0};
  constexpr auto sBigMax = std::numeric_limits<SIntT>::max();

  constexpr auto uZero   = UIntT{0};
  constexpr auto uBigMax = std::numeric_limits<UIntT>::max();

  // Constants to avoid casting in place

  constexpr auto O_C  = static_cast<signed char>(0);
  constexpr auto O_UC = static_cast<unsigned char>(0);

  constexpr auto O_S  = static_cast<signed short int>(0);
  constexpr auto O_US = static_cast<unsigned short int>(0);

  // signed char

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name",
  // here and below...
  { [[maybe_unused]] std::same_as<signed char> decltype(auto) _ = std::saturate_cast<signed char>(SCHAR_MAX); }
  assert(std::saturate_cast<signed char>(SCHAR_MIN)  == SCHAR_MIN);
  assert(std::saturate_cast<signed char>(      O_C)  ==       O_C);
  assert(std::saturate_cast<signed char>(SCHAR_MAX)  == SCHAR_MAX);

  { [[maybe_unused]] std::same_as<signed char> decltype(auto) _ = std::saturate_cast<signed char>(UCHAR_MAX); }
  assert(std::saturate_cast<signed char>(     O_UC)  ==       O_C);
  assert(std::saturate_cast<signed char>(UCHAR_MAX)  == SCHAR_MAX);

  { [[maybe_unused]] std::same_as<signed char> decltype(auto) _ = std::saturate_cast<signed char>(sBigMax); }
  assert(std::saturate_cast<signed char>(sBigMin)    == SCHAR_MIN); // saturated
  assert(std::saturate_cast<signed char>(  sZero)    ==       O_C);
  assert(std::saturate_cast<signed char>(sBigMax)    == SCHAR_MAX); // saturated

  { [[maybe_unused]] std::same_as<signed char> decltype(auto) _ = std::saturate_cast<signed char>(uBigMax); }
  assert(std::saturate_cast<signed char>(  uZero)    ==       O_C);
  assert(std::saturate_cast<signed char>(uBigMax)    == SCHAR_MAX); // saturated

  // short

  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(SCHAR_MAX); }
  assert(std::saturate_cast<signed short int>(SCHAR_MIN) == static_cast<signed short int>(SCHAR_MIN));
  assert(std::saturate_cast<signed short int>(      O_C) == O_S);
  assert(std::saturate_cast<signed short int>(SCHAR_MAX) == static_cast<signed short int>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(UCHAR_MAX); }
  assert(std::saturate_cast<signed short int>(     O_UC) == O_S);
  assert(std::saturate_cast<signed short int>(UCHAR_MAX) == static_cast<signed short int>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(SHRT_MAX); }
  assert(std::saturate_cast<signed short int>( SHRT_MIN) == SHRT_MIN);
  assert(std::saturate_cast<signed short int>(      O_S) == O_S);
  assert(std::saturate_cast<signed short int>( SHRT_MAX) == SHRT_MAX);

  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(USHRT_MAX); }
  assert(std::saturate_cast<signed short int>(     O_US) == O_S);
  assert(std::saturate_cast<signed short int>(USHRT_MAX) == SHRT_MAX); // saturated

  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(sBigMax); }
  assert(std::saturate_cast<signed short int>( sBigMin)   == SHRT_MIN); // saturated
  assert(std::saturate_cast<signed short int>(   sZero)   == O_S);
  assert(std::saturate_cast<signed short int>( sBigMax)   == SHRT_MAX); // saturated

  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(uBigMax); }
  assert(std::saturate_cast<signed short int>(   uZero)   == O_S);
  assert(std::saturate_cast<signed short int>( uBigMax)   == SHRT_MAX); // saturated

  // int

  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(SCHAR_MAX); }
  assert(std::saturate_cast<signed int>(SCHAR_MIN) == static_cast<signed int>(SCHAR_MIN));
  assert(std::saturate_cast<signed int>(      O_C) == 0);
  assert(std::saturate_cast<signed int>(SCHAR_MAX) == static_cast<signed int>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(UCHAR_MAX); }
  assert(std::saturate_cast<signed int>(     O_UC) == 0);
  assert(std::saturate_cast<signed int>(UCHAR_MAX) == static_cast<signed int>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(INT_MAX); }
  assert(std::saturate_cast<signed int>(  INT_MIN) == INT_MIN);
  assert(std::saturate_cast<signed int>(        0) == 0);
  assert(std::saturate_cast<signed int>(  INT_MAX) == INT_MAX);

  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(UINT_MAX); }
  assert(std::saturate_cast<signed int>(       0)  == 0);
  assert(std::saturate_cast<signed int>(UINT_MAX)  == INT_MAX); // saturated

  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(sBigMax); }
  assert(std::saturate_cast<signed int>( sBigMin)  == INT_MIN); // saturated
  assert(std::saturate_cast<signed int>(   sZero)  == 0);
  assert(std::saturate_cast<signed int>( sBigMax)  == INT_MAX); // saturated

  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(uBigMax); }
  assert(std::saturate_cast<signed int>( uZero)    == 0);
  assert(std::saturate_cast<signed int>( uBigMax)  == INT_MAX); // saturated

  // long

  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(SCHAR_MAX); }
  assert(std::saturate_cast<signed long int>(SCHAR_MIN) == static_cast<signed long int>(SCHAR_MIN));
  assert(std::saturate_cast<signed long int>(      O_C) == 0L);
  assert(std::saturate_cast<signed long int>(SCHAR_MAX) == static_cast<signed long int>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(UCHAR_MAX); }
  assert(std::saturate_cast<signed long int>(     O_UC) == 0L);
  assert(std::saturate_cast<signed long int>(UCHAR_MAX) == static_cast<signed long int>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(LONG_MAX); }
  assert(std::saturate_cast<signed long int>( LONG_MIN) == LONG_MIN);
  assert(std::saturate_cast<signed long int>(       0L) == 0L);
  assert(std::saturate_cast<signed long int>( LONG_MAX) == LONG_MAX);

  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(ULONG_MAX); }
  assert(std::saturate_cast<signed long int>(      0UL) == 0L);
  assert(std::saturate_cast<signed long int>(ULONG_MAX) == LONG_MAX); // saturated

  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(sBigMax); }
  assert(std::saturate_cast<signed long int>(  sBigMin) == LONG_MIN); // saturated
  assert(std::saturate_cast<signed long int>(    sZero) == 0L);
  assert(std::saturate_cast<signed long int>(  sBigMax) == LONG_MAX); // saturated

  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(uBigMax); }
  assert(std::saturate_cast<signed long int>(    uZero) == 0L);
  assert(std::saturate_cast<signed long int>(  uBigMax) == LONG_MAX); // saturated

  // long long

  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(SCHAR_MAX); }
  assert(std::saturate_cast<signed long long int>(SCHAR_MIN) == static_cast<signed long long int>(SCHAR_MIN));
  assert(std::saturate_cast<signed long long int>(      0LL) == 0LL);
  assert(std::saturate_cast<signed long long int>(SCHAR_MAX) == static_cast<signed long long int>(SCHAR_MAX));

  { [[maybe_unused]]   std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(UCHAR_MAX); }
  assert(std::saturate_cast<signed long long int>(     O_UC) == 0LL);
  assert(std::saturate_cast<signed long long int>(UCHAR_MAX) == static_cast<signed long long int>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(LLONG_MIN); }
  assert(std::saturate_cast<signed long long int>(LLONG_MIN) == LLONG_MIN);
  assert(std::saturate_cast<signed long long int>(      0LL) == 0LL);
  assert(std::saturate_cast<signed long long int>(LLONG_MAX) == LLONG_MAX);

  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(ULLONG_MAX); }
  assert(std::saturate_cast<signed long long int>(      0ULL) == 0LL);
  assert(std::saturate_cast<signed long long int>(ULLONG_MAX) == LLONG_MAX); // saturated

#ifndef TEST_HAS_NO_INT128
  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(sBigMax); }
  assert(std::saturate_cast<signed long long int>(   sBigMin) == LLONG_MIN); // (128-bit) saturated
  assert(std::saturate_cast<signed long long int>(     sZero) == 0LL);
  assert(std::saturate_cast<signed long long int>(   sBigMax) == LLONG_MAX); // (128-bit) saturated

  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(uBigMax); }
  assert(std::saturate_cast<signed long long int>(     uZero) == 0LL);
  assert(std::saturate_cast<signed long long int>(   uBigMax) == LLONG_MAX); // (128-bit) saturated

  { [[maybe_unused]] std::same_as<__int128_t> decltype(auto) _ = std::saturate_cast<__int128_t>(SCHAR_MAX); }
  assert(std::saturate_cast<__int128_t>(SCHAR_MIN) == static_cast<__int128_t>(SCHAR_MIN));
  assert(std::saturate_cast<__int128_t>(      O_C) == sZero);
  assert(std::saturate_cast<__int128_t>(SCHAR_MAX) == static_cast<__int128_t>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<__int128_t> decltype(auto) _ = std::saturate_cast<__int128_t>(UCHAR_MAX); }
  assert(std::saturate_cast<__int128_t>(     O_UC) == sZero);
  assert(std::saturate_cast<__int128_t>(UCHAR_MAX) == static_cast<__int128_t>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<__int128_t> decltype(auto) _ = std::saturate_cast<__int128_t>(sBigMax); }
  assert(std::saturate_cast<__int128_t>(  sBigMin) == sBigMin);
  assert(std::saturate_cast<__int128_t>(    sZero) == sZero);
  assert(std::saturate_cast<__int128_t>(  sBigMax) == sBigMax);

  { [[maybe_unused]] std::same_as<__int128_t> decltype(auto) _ = std::saturate_cast<__int128_t>(uBigMax); }
  assert(std::saturate_cast<__int128_t>(    uZero) == sZero);
  assert(std::saturate_cast<__int128_t>(  uBigMax) == sBigMax); // saturated
#endif

  // unsigned char

  { [[maybe_unused]] std::same_as<unsigned char> decltype(auto) _ = std::saturate_cast<unsigned char>(SCHAR_MAX); }
  assert(std::saturate_cast<unsigned char>(SCHAR_MIN) == O_UC);
  assert(std::saturate_cast<unsigned char>(      O_C) == O_UC);
  assert(std::saturate_cast<unsigned char>(SCHAR_MAX) == static_cast<unsigned char>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned char> decltype(auto) _ = std::saturate_cast<unsigned char>(UCHAR_MAX); }
  assert(std::saturate_cast<unsigned char>(     O_UC) == O_UC);
  assert(std::saturate_cast<unsigned char>(UCHAR_MAX) == UCHAR_MAX);

  { [[maybe_unused]] std::same_as<unsigned char> decltype(auto) _ = std::saturate_cast<unsigned char>(sBigMax); }
  assert(std::saturate_cast<unsigned char>(  sBigMin) == O_UC);      // saturated
  assert(std::saturate_cast<unsigned char>(    sZero) == O_UC);
  assert(std::saturate_cast<unsigned char>(  sBigMax) == UCHAR_MAX); // saturated

  { [[maybe_unused]] std::same_as<unsigned char> decltype(auto) _ = std::saturate_cast<unsigned char>(uBigMax); }
  assert(std::saturate_cast<unsigned char>(    uZero) == O_UC);
  assert(std::saturate_cast<unsigned char>(  uBigMax) == UCHAR_MAX); // saturated

  // unsigned short

  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(SCHAR_MAX); }
  assert(std::saturate_cast<unsigned short int>(SCHAR_MIN) == O_US);
  assert(std::saturate_cast<unsigned short int>(      O_C) == O_US);
  assert(std::saturate_cast<unsigned short int>(SCHAR_MAX) == static_cast<unsigned short int>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(UCHAR_MAX); }
  assert(std::saturate_cast<unsigned short int>(     O_UC) == O_US);
  assert(std::saturate_cast<unsigned short int>(UCHAR_MAX) == static_cast<unsigned short int>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(SCHAR_MIN); }
  assert(std::saturate_cast<unsigned short int>( SHRT_MIN) == O_US);
  assert(std::saturate_cast<unsigned short int>(      O_S) == O_US);
  assert(std::saturate_cast<unsigned short int>( SHRT_MAX) == static_cast<unsigned short int>(SHRT_MAX));

  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(UCHAR_MAX); }
  assert(std::saturate_cast<unsigned short int>(     O_US) == O_US);
  assert(std::saturate_cast<unsigned short int>(USHRT_MAX) == USHRT_MAX);

  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(sBigMax); }
  assert(std::saturate_cast<unsigned short int>(  sBigMin) == O_US);      // saturated
  assert(std::saturate_cast<unsigned short int>(    sZero) == O_US);
  assert(std::saturate_cast<unsigned short int>(  sBigMax) == USHRT_MAX); // saturated

  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(uBigMax); }
  assert(std::saturate_cast<unsigned short int>(    uZero) == O_US);
  assert(std::saturate_cast<unsigned short int>(  uBigMax) == USHRT_MAX); // saturated

  // unsigned int

  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(SCHAR_MAX); }
  assert(std::saturate_cast<unsigned int>(SCHAR_MIN) == O_US);
  assert(std::saturate_cast<unsigned int>(     O_UC) == 0U);
  assert(std::saturate_cast<unsigned int>(SCHAR_MAX) == static_cast<unsigned int>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(UCHAR_MAX); }
  assert(std::saturate_cast<unsigned int>(     O_UC) == 0U);
  assert(std::saturate_cast<unsigned int>(UCHAR_MAX) == static_cast<unsigned int>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(INT_MAX); }
  assert(std::saturate_cast<unsigned int>(  INT_MIN) == 0U);
  assert(std::saturate_cast<unsigned int>(        0) == 0U);
  assert(std::saturate_cast<unsigned int>(  INT_MAX) == static_cast<unsigned int>(INT_MAX));

  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(UINT_MAX); }
  assert(std::saturate_cast<unsigned int>(       0U) == 0U);
  assert(std::saturate_cast<unsigned int>( UINT_MAX) == UINT_MAX);

  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(sBigMax); }
  assert(std::saturate_cast<unsigned int>(  sBigMin) == 0U);       // saturated
  assert(std::saturate_cast<unsigned int>(    sZero) == 0U);
  assert(std::saturate_cast<unsigned int>(  sBigMax) == UINT_MAX); // saturated
  
  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(uBigMax); }
  assert(std::saturate_cast<unsigned int>(    uZero) == 0U);
  assert(std::saturate_cast<unsigned int>(  uBigMax) == UINT_MAX);  // saturated

  // unsigned long

  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(SCHAR_MAX); }
  assert(std::saturate_cast<unsigned long int>(SCHAR_MIN) == 0UL);
  assert(std::saturate_cast<unsigned long int>(      O_C) == 0UL);
  assert(std::saturate_cast<unsigned long int>(SCHAR_MAX) == static_cast<unsigned long int>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(UCHAR_MAX); }
  assert(std::saturate_cast<unsigned long int>(     O_UC) == 0UL);
  assert(std::saturate_cast<unsigned long int>(UCHAR_MAX) == static_cast<unsigned long int>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(LONG_MAX); }
  assert(std::saturate_cast<unsigned long int>( LONG_MIN) == 0UL);
  assert(std::saturate_cast<unsigned long int>(       0L) == 0UL);
  assert(std::saturate_cast<unsigned long int>( LONG_MAX) == static_cast<unsigned long int>(LONG_MAX));

  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(ULONG_MAX); }
  assert(std::saturate_cast<unsigned long int>(      0UL) == 0UL);
  assert(std::saturate_cast<unsigned long int>(ULONG_MAX) == ULONG_MAX);

  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(sBigMax); }
  assert(std::saturate_cast<unsigned long int>(  sBigMin) == 0UL);       // saturated
  assert(std::saturate_cast<unsigned long int>(    sZero) == 0UL);
  assert(std::saturate_cast<unsigned long int>(  sBigMax) == ULONG_MAX); // saturated

  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(uBigMax); }
  assert(std::saturate_cast<unsigned long int>(    uZero) == 0UL);
  assert(std::saturate_cast<unsigned long int>(  uBigMax) == ULONG_MAX); // saturated

  // unsigned long long

  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(SCHAR_MAX); }
  assert(std::saturate_cast<unsigned long long int>( SCHAR_MIN) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>(       O_C) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>( SCHAR_MAX) == static_cast<unsigned long long int>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned long long  int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(UCHAR_MAX); }
  assert(std::saturate_cast<unsigned long long int>(      O_UC) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>( UCHAR_MAX) == static_cast<unsigned long long int>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(LLONG_MAX); }
  assert(std::saturate_cast<unsigned long long int>( LLONG_MIN) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>(       0LL) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>( LLONG_MAX) == static_cast<unsigned long long int>(LLONG_MAX));

  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(ULLONG_MAX); }
  assert(std::saturate_cast<unsigned long long int>(      0ULL) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>(ULLONG_MAX) == ULLONG_MAX);

#ifndef TEST_HAS_NO_INT128
  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(sBigMax); }
  assert(std::saturate_cast<unsigned long long int>(   sBigMin) == 0ULL);       // (128-bit) saturated
  assert(std::saturate_cast<unsigned long long int>(     sZero) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>(   sBigMax) == ULLONG_MAX); // (128-bit) saturated

  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(uBigMax); }
  assert(std::saturate_cast<unsigned long long int>(     uZero) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>(   uBigMax) == ULLONG_MAX); // (128-bit) saturated

  { [[maybe_unused]] std::same_as<__uint128_t> decltype(auto) _ = std::saturate_cast<__uint128_t>(SCHAR_MIN); }
  assert(std::saturate_cast<__uint128_t>(SCHAR_MIN) == uZero);
  assert(std::saturate_cast<__uint128_t>(      O_C) == uZero);
  assert(std::saturate_cast<__uint128_t>(SCHAR_MAX) == static_cast<__uint128_t>(SCHAR_MAX));

  { [[maybe_unused]] std::same_as<__uint128_t> decltype(auto) _ = std::saturate_cast<__uint128_t>(UCHAR_MAX); }
  assert(std::saturate_cast<__uint128_t>(     O_UC) == uZero);
  assert(std::saturate_cast<__uint128_t>(UCHAR_MAX) == static_cast<__uint128_t>(UCHAR_MAX));

  { [[maybe_unused]] std::same_as<__uint128_t> decltype(auto) _ = std::saturate_cast<__uint128_t>(sBigMax); }
  assert(std::saturate_cast<__uint128_t>(  sBigMin) == uZero); // saturated
  assert(std::saturate_cast<__uint128_t>(    sZero) == uZero);
  assert(std::saturate_cast<__uint128_t>(  sBigMax) == static_cast<__uint128_t>(sBigMax));

  { [[maybe_unused]] std::same_as<__uint128_t> decltype(auto) _ = std::saturate_cast<__uint128_t>(uBigMax); }
  assert(std::saturate_cast<__uint128_t>(    uZero) == uZero);
  assert(std::saturate_cast<__uint128_t>(  uBigMax) == uBigMax);
#endif

  // clang-format on

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
