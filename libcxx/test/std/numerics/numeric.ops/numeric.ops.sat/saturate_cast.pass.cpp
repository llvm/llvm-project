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

// Larger to smaller
static_assert(noexcept(std::saturate_cast<signed char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(std::saturate_cast<signed char>(std::numeric_limits<unsigned int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned char>(std::numeric_limits<unsigned int>::max())));
// Smaller to larger
static_assert(noexcept(std::saturate_cast<signed int>(std::numeric_limits<signed char>::max())));
static_assert(noexcept(std::saturate_cast<signed int>(std::numeric_limits<unsigned char>::max())));
static_assert(noexcept(std::saturate_cast<unsigned int>(std::numeric_limits<signed char>::max())));
static_assert(noexcept(std::saturate_cast<unsigned int>(std::numeric_limits<unsigned char>::max())));
// Same type
static_assert(noexcept(std::saturate_cast<signed long int>(std::numeric_limits<signed long int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned long int>(std::numeric_limits<unsigned long int>::max())));

constexpr bool test() {
#ifndef TEST_HAS_NO_INT128
  using SIntT = __int128_t;
  using UIntT = __uint128_t;
#else
  using SIntT = long long int;
  using UIntT = unsigned long long int;
#endif

  // Constants: biggest numbers

  constexpr auto sintMin  = std::numeric_limits<SIntT>::min();
  constexpr auto sintZero = SIntT{0};
  constexpr auto sintMax  = std::numeric_limits<SIntT>::max();

  constexpr auto uintMin  = std::numeric_limits<UIntT>::min();
  constexpr auto uintZero = UIntT{0};
  constexpr auto uintMax  = std::numeric_limits<UIntT>::max();

  // Constants: numeric limits

  constexpr auto std_scharMin  = std::numeric_limits<signed char>::min();
  constexpr auto std_scharZero = static_cast<signed char>(0);
  constexpr auto std_scharMax  = std::numeric_limits<signed char>::max();

  constexpr auto std_ucharMin  = std::numeric_limits<unsigned char>::min();
  constexpr auto std_ucharZero = static_cast<unsigned char>(0);
  constexpr auto std_ucharMax  = std::numeric_limits<unsigned char>::max();

  constexpr auto std_ssintMin  = std::numeric_limits<signed short int>::min();
  constexpr auto std_ssintZero = static_cast<signed short int>(0);
  constexpr auto std_ssintMax  = std::numeric_limits<signed short int>::max();

  constexpr auto std_usintMin  = std::numeric_limits<unsigned short int>::min();
  constexpr auto std_usintZero = static_cast<unsigned short int>(0);
  constexpr auto std_usintMax  = std::numeric_limits<unsigned short int>::max();

  // constexpr auto std_sintMin = std::numeric_limits<signed int>::min();
  // constexpr auto std_sintMax = std::numeric_limits<signed int>::max();

  // constexpr auto std_uintMin = std::numeric_limits<unsigned int>::min();
  // constexpr auto std_uintMax = std::numeric_limits<unsigned int>::max();

  constexpr auto std_slMin = std::numeric_limits<signed long int>::min();
  constexpr auto std_slMax = std::numeric_limits<signed long int>::max();

  // constexpr auto std_ulMin = std::numeric_limits<unsigned long int>::min();
  constexpr auto std_ulMax = std::numeric_limits<unsigned long int>::max();

  constexpr auto std_sllMin = std::numeric_limits<signed long long int>::min();
  constexpr auto std_sllMax = std::numeric_limits<signed long long int>::max();

  // constexpr auto std_ullMin = std::numeric_limits<unsigned long long int>::min();
  constexpr auto std_ullMax = std::numeric_limits<unsigned long long int>::max();

  // clang-format off
  
  // signed char

  assert(std::saturate_cast<signed char>(std_scharMin)  == std_scharMin);
  assert(std::saturate_cast<signed char>(std_scharZero) == std_scharZero);
  assert(std::saturate_cast<signed char>(std_scharMax)  == std_scharMax);

  assert(std::saturate_cast<signed char>(std_ucharMin)  == std_scharZero);
  assert(std::saturate_cast<signed char>(std_ucharZero) == std_scharZero);
  assert(std::saturate_cast<signed char>(std_ucharMax)  == std_scharMax);

  assert(std::saturate_cast<signed char>(sintMin)       == std_scharMin);  // saturated
  assert(std::saturate_cast<signed char>(sintZero)      == std_scharZero);
  assert(std::saturate_cast<signed char>(sintMax)       == std_scharMax);  // saturated

  assert(std::saturate_cast<signed char>(uintMin)       == std_scharZero);
  assert(std::saturate_cast<signed char>(uintZero)      == std_scharZero);
  assert(std::saturate_cast<signed char>(uintMax)       == std_scharMax);  // saturated

  // short

  std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(std_scharMin);
  assert(std::saturate_cast<signed short int>(std_scharMin)  == static_cast<signed short int>(std_scharMin));
  assert(std::saturate_cast<signed short int>(std_scharZero) == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_scharMax)  == static_cast<signed short int>(std_scharMax));

  std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(std_ucharMin);
  assert(std::saturate_cast<signed short int>(std_ucharMin)  == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_ucharZero) == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_ucharMax)  == static_cast<signed short int>(std_ucharMax));

  std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(std_ssintMin);
  assert(std::saturate_cast<signed short int>(std_ssintMin)  == std_ssintMin);
  assert(std::saturate_cast<signed short int>(std_ssintZero) == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_ssintMax)  == std_ssintMax);

  std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(std_usintMin);
  assert(std::saturate_cast<signed short int>(std_usintMin)  == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_usintZero) == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_usintMax)  == std_ssintMax);  // saturated

  std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(sintMin);
  assert(std::saturate_cast<signed short int>(sintMin)       == std_ssintMin);  // saturated
  assert(std::saturate_cast<signed short int>(sintZero)      == std_ssintZero);
  assert(std::saturate_cast<signed short int>(sintMax)       == std_ssintMax);  // saturated

  std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(uintMin);
  assert(std::saturate_cast<signed short int>(uintMin)       == std_ssintZero);
  assert(std::saturate_cast<signed short int>(uintZero)      == std_ssintZero);
  assert(std::saturate_cast<signed short int>(uintMax)       == std_ssintMax);  // saturated

  // int

  assert(std::saturate_cast<int>(sintMin) == -2'147'483'648);
  assert(std::saturate_cast<int>(-2'147'483'648) == -2'147'483'648);
  assert(std::saturate_cast<int>(0) == 0);
  assert(std::saturate_cast<int>(2'147'483'647) == 2'147'483'647);
  assert(std::saturate_cast<int>(sintMax) == 2'147'483'647);

  // long

  assert(std::saturate_cast<long int>(sintMin) == std_slMin);
  // assert(std::saturate_cast<long int>(-2'147'483'648L) == std_slMin);
  assert(std::saturate_cast<long int>(0L) == 0L);
  // assert(std::saturate_cast<long int>(2'147'483'647L) == std_slMax);
  assert(std::saturate_cast<long int>(sintMax) == std_slMax);

  // long long

  assert(std::saturate_cast<long long int>(sintMin) == std_sllMin);
  assert(std::saturate_cast<long long int>(sintMin) == std_sllMin);
  // assert(std::saturate_cast<long long int>(-9'223'372'036'854'775'808LL) == std_sllMin);
  assert(std::saturate_cast<long long int>(0LL) == 0LL);
  assert(std::saturate_cast<long long int>(9'223'372'036'854'775'807LL) == std_sllMax);
  assert(std::saturate_cast<long long int>(sintMax) == std_sllMax);

#ifndef TEST_HAS_NO_INT128
  assert(std::saturate_cast<__int128_t>(sintMin)  == sintMin);
  assert(std::saturate_cast<__int128_t>(sintZero) == sintZero);
  assert(std::saturate_cast<__int128_t>(sintMax)  == sintMax);
#endif

  // unsigned char

  assert(std::saturate_cast<unsigned char>(std_scharMin)  == std_ucharMin);
  assert(std::saturate_cast<unsigned char>(std_scharZero) == std_ucharZero);
  assert(std::saturate_cast<unsigned char>(std_scharMax)  == static_cast<unsigned char>(std_scharMax));
  assert(std::saturate_cast<unsigned char>(std_ucharMin)  == std_ucharMin);
  assert(std::saturate_cast<unsigned char>(std_ucharZero) == std_ucharZero);
  assert(std::saturate_cast<unsigned char>(std_ucharMax)  == std_ucharMax);
  assert(std::saturate_cast<unsigned char>(sintMin)       == std_ucharMin);  // saturated
  assert(std::saturate_cast<unsigned char>(sintZero)      == std_ucharZero);
  assert(std::saturate_cast<unsigned char>(sintMax)       == std_ucharMax);  // saturated
  assert(std::saturate_cast<unsigned char>(uintMin)       == std_ucharMin);
  assert(std::saturate_cast<unsigned char>(uintZero)      == std_ucharZero);
  assert(std::saturate_cast<unsigned char>(uintMax)       == std_ucharMax);  // saturated

  // short

  assert(std::saturate_cast<unsigned short int>(0) == 0);
  assert(std::saturate_cast<unsigned short int>(65'535) == 65'535);
  assert(std::saturate_cast<unsigned short int>(uintMax) == 65'535);

  // unsigned int

  assert(std::saturate_cast<unsigned int>(0U) == 0U);
  assert(std::saturate_cast<unsigned int>(4'294'967'295U) == 4'294'967'295U);
  assert(std::saturate_cast<unsigned int>(uintMax) == 4'294'967'295U);

  // unsigned long

  assert(std::saturate_cast<unsigned long int>(0UL) == 0UL);
  // assert(std::saturate_cast<unsigned long int>(4'294'967'295UL) == std_ulMax);
  assert(std::saturate_cast<unsigned long int>(uintMax) == std_ulMax);

  // unsigned long long

  assert(std::saturate_cast<unsigned long long int>(0ULL) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>(18'446'744'073'709'551'615ULL) == std_ullMax);
  assert(std::saturate_cast<unsigned long long int>(uintMax) == std_ullMax);

#ifndef TEST_HAS_NO_INT128
  assert(std::saturate_cast<__uint128_t>(uintMin)  == uintMin);
  assert(std::saturate_cast<__uint128_t>(uintZero) == uintZero);
  assert(std::saturate_cast<__uint128_t>(uintMax)  == uintMax);
#endif

  // clang-format on

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
