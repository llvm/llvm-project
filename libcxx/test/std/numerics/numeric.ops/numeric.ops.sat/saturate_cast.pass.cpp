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
#include <limits>
#include <numeric>

static_assert(noexcept(std::saturate_cast<signed char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(std::saturate_cast<signed char>(std::numeric_limits<unsigned int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned char>(std::numeric_limits<unsigned int>::max())));

#include <print>

// constexpr
bool test() {
  // clang-format off

#ifndef TEST_HAS_NO_INT128
  using SIntT = __int128_t;
  using UIntT = __uint128_t;
#else
  using SIntT = long long int;
  using UIntT = unsigned long long int;
#endif

  // Biggest numbers

  constexpr auto sintMin = std::numeric_limits<SIntT>::min();
  constexpr auto sintMax = std::numeric_limits<SIntT>::max();
  constexpr auto sintZero = SIntT{0};

  constexpr auto uintMin = std::numeric_limits<UIntT>::min();
  constexpr auto uintMax = std::numeric_limits<UIntT>::max();
  constexpr auto uintZero = UIntT{0};

  // Limits

  // constexpr auto std_scharMin = std::numeric_limits<signed char>::min();
  // constexpr auto std_scharMax = std::numeric_limits<signed char>::max();

  // constexpr auto std_ucharMin = std::numeric_limits<unsigned char>::min();
  // constexpr auto std_ucharMax = std::numeric_limits<unsigned char>::max();

  // constexpr auto std_ssintMin = std::numeric_limits<signed short int>::min();
  // constexpr auto std_ssintMax = std::numeric_limits<unsigned short int>::max();

  // constexpr auto std_usintMin = std::numeric_limits<signed short int>::min();
  // constexpr auto std_usintMax = std::numeric_limits<unsigned short int>::max();

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

  // signed char: -128 to 127

  assert(std::saturate_cast<signed char>(sintMin) == -128);
  assert(std::saturate_cast<signed char>(-128) == -128);
  assert(std::saturate_cast<signed char>(0) == 0);
  assert(std::saturate_cast<signed char>(127) == 127);
  assert(std::saturate_cast<signed char>(sintMax) == 127);

  // short: -32,768 to 32,767

  // std::println(stderr, "results: {}", std::saturate_cast<short int>(sintMin));
  assert(std::saturate_cast<short int>(sintMin) == -32'768);
  assert(std::saturate_cast<short int>(-32'768) == -32'768);
  assert(std::saturate_cast<short int>(0) == 0);
  assert(std::saturate_cast<short int>(32'767) == 32'767);
  assert(std::saturate_cast<short int>(sintMax) == 32'767);

  // int: -2,147,483,648 to 2,147,483,647

  assert(std::saturate_cast<int>(sintMin) == -2'147'483'648);
  assert(std::saturate_cast<int>(-2'147'483'648) == -2'147'483'648);
  assert(std::saturate_cast<int>(0) == 0);
  assert(std::saturate_cast<int>(2'147'483'647) == 2'147'483'647);
  assert(std::saturate_cast<int>(sintMax) == 2'147'483'647);

  // long: -2,147,483,648 to 2,147,483,647

  assert(std::saturate_cast<long int>(sintMin) == std_slMin);
  // assert(std::saturate_cast<long int>(-2'147'483'648L) == std_slMin);
  assert(std::saturate_cast<long int>(0L) == 0L);
  // assert(std::saturate_cast<long int>(2'147'483'647L) == std_slMax);
  assert(std::saturate_cast<long int>(sintMax) == std_slMax);

  // long long: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807

  assert(std::saturate_cast<long long int>(sintMin) == std_sllMin);
  assert(std::saturate_cast<long long int>(sintMin) == -std_sllMin);
  // assert(std::saturate_cast<long long int>(-9'223'372'036'854'775'808LL) == std_sllMin);
  assert(std::saturate_cast<long long int>(0LL) == 0LL);
  assert(std::saturate_cast<long long int>(9'223'372'036'854'775'807LL) == std_sllMax);
  assert(std::saturate_cast<long long int>(sintMax) == std_sllMax);

#ifndef TEST_HAS_NO_INT128
  assert(std::saturate_cast<__int128_t>(sintMin) == sintMin);
  assert(std::saturate_cast<__int128_t>(sintZero) == sintZero);
  assert(std::saturate_cast<__int128_t>(sintMax) == sintMax);
#endif

  // unsigned char: 0 to 255

  assert(std::saturate_cast<unsigned char>(0) == 0);
  assert(std::saturate_cast<unsigned char>(127) == 127);
  assert(std::saturate_cast<unsigned char>(uintMax) == 255);

  // short: 0 to 65,535

  assert(std::saturate_cast<unsigned short int>(0) == 0);
  assert(std::saturate_cast<unsigned short int>(65'535) == 65'535);
  assert(std::saturate_cast<unsigned short int>(uintMax) == 65'535);

  // unsigned int: 0 to 4,294,967,295

  assert(std::saturate_cast<unsigned int>(0U) == 0U);
  assert(std::saturate_cast<unsigned int>(4'294'967'295U) == 4'294'967'295U);
  assert(std::saturate_cast<unsigned int>(uintMax) == 4'294'967'295U);

  // unsigned long: 0 to 4,294,967,295

  assert(std::saturate_cast<unsigned long int>(0UL) == 0UL);
  // assert(std::saturate_cast<unsigned long int>(4'294'967'295UL) == std_ulMax);
  assert(std::saturate_cast<unsigned long int>(uintMax) == std_ulMax);

  // unsigned long long: 0 to 18,446,744,073,709,551,615

  assert(std::saturate_cast<unsigned long long int>(0ULL) == 0ULL);
  assert(std::saturate_cast<unsigned long long int>(18'446'744'073'709'551'615ULL) == std_ullMax);
  assert(std::saturate_cast<unsigned long long int>(uintMax) == std_ullMax);

#ifndef TEST_HAS_NO_INT128
  assert(std::saturate_cast<__uint128_t>(uintMin) == uintMin);
  assert(std::saturate_cast<__uint128_t>(uintZero) == uintZero);
  assert(std::saturate_cast<__uint128_t>(uintMax) == uintMax);
#endif

  // clang-format on

  return true;
}

int main(int, char**) {
  test();
  // static_assert(test());

  return 0;
}
