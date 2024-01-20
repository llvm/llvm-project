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

static_assert(noexcept(saturate_cast<signed char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(saturate_cast<signed char>(std::numeric_limits<unsigned int>::max())));
static_assert(noexcept(saturate_cast<unsigned char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(saturate_cast<unsigned char>(std::numeric_limits<unsigned int>::max())));

constexpr bool test_signed() {
  // clang-format on

#ifndef TEST_HAS_NO_INT128
  using BiggestSIntT = __int128_t;
#else
  using BiggestSIntT = long long int;
#endif

  // signed char: -128 to 127

  assert(std::saturate_cast<signed char, BiggestSIntT>(-255) == -128);
  assert(std::saturate_cast<signed char, BiggestSIntT>(-128) == -128);
  assert(std::saturate_cast<signed char, BiggestSIntT>(0) == 0);
  assert(std::saturate_cast<signed char, BiggestSIntT>(127) == 127);
  assert(std::saturate_cast<signed char, BiggestSIntT>(255) == 127);

  // short: -32,768 to 32,767

  assert(std::saturate_cast<short int, BiggestSIntT>(-255) == -128);
  assert(std::saturate_cast<short int, BiggestSIntT>(-32'768) == -32'767);
  assert(std::saturate_cast<short int, BiggestSIntT>(0) == 0);
  assert(std::saturate_cast<short int, BiggestSIntT>(32'768) == 32'768);
  assert(std::saturate_cast<short int, BiggestSIntT>(255) == 32'768);

  // int: -2,147,483,648 to 2,147,483,647

  assert(std::saturate_cast<int, BiggestSIntT>(-255) == -2'147'483'648);
  assert(std::saturate_cast<int, BiggestSIntT>(-2'147'483'648) == -2'147'483'648);
  assert(std::saturate_cast<int, BiggestSIntT>(0) == 0);
  assert(std::saturate_cast<int, BiggestSIntT>(2'147'483'647) == 2'147'483'647);
  assert(std::saturate_cast<int, BiggestSIntT>(255) == 2'147'483'647);

  // long: -2,147,483,648 to 2,147,483,647

  assert(std::saturate_cast<long int, BiggestSIntT>(-255) == -2'147'483'648);
  assert(std::saturate_cast<long int, BiggestSIntT>(-2'147'483'648) == -2'147'483'648);
  assert(std::saturate_cast<long int, BiggestSIntT>(0) == 0);
  assert(std::saturate_cast<long int, BiggestSIntT>(2'147'483'647) == 2'147'483'647);
  assert(std::saturate_cast<long int, BiggestSIntT>(255) == 2'147'483'647);

  // long long: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807

  assert(std::saturate_cast<long long int, BiggestSIntT>(-255) == -9'223'372'036'854'775'808);
  assert(std::saturate_cast<long long int, BiggestSIntT>(-9'223'372'036'854'775'808) == -9'223'372'036'854'775'808);
  assert(std::saturate_cast<long long int, BiggestSIntT>(0) == 0);
  assert(std::saturate_cast<long long int, BiggestSIntT>(9'223'372'036'854'775'807) == 9'223'372'036'854'775'807);
  assert(std::saturate_cast<long long int, BiggestSIntT>(255) == 9'223'372'036'854'775'807);

#ifndef TEST_HAS_NO_INT128
  constexpr auto int128min = std::numeric_limits<__int128_t>::min();
  constexpr auto int128max = std::numeric_limits<__int128_t>::max();

  assert(std::saturate_cast<long long int, BiggestSIntT>(int128min) == -9'223'372'036'854'775'808);
  assert(std::saturate_cast<long long int, BiggestSIntT>(-9'223'372'036'854'775'808) == -9'223'372'036'854'775'808);
  assert(std::saturate_cast<long long int, BiggestSIntT>(0) == 0);
  assert(std::saturate_cast<long long int, BiggestSIntT>(9'223'372'036'854'775'807) == 9'223'372'036'854'775'807);
  assert(std::saturate_cast<long long int, BiggestSIntT>(int128max) == 9'223'372'036'854'775'807);

  assert(std::saturate_cast<__int128_t, BiggestSIntT>(int128min) == int128min);
  assert(std::saturate_cast<__int128_t, BiggestSIntT>(0) == 0);
  assert(std::saturate_cast<__int128_t, BiggestSIntT>(int128max) == int128max);
#endif
  // clang-format off
}

constexpr void test_unsigned()
{
  // clang-format off

#ifndef TEST_HAS_NO_INT128
  using BiggestUIntT = __uint128_t;
#else 
  using BiggestUIntT = unsigned long long int;
#endif

  // unsigned char: 0 to 255

  assert(std::saturate_cast<unsigned char, BiggestUIntT>(0) == 0);
  assert(std::saturate_cast<unsigned char, BiggestUIntT>(127) == 127);
  assert(std::saturate_cast<unsigned char, BiggestUIntT>(255) == 127);

  // short: 0 to 65,535

  assert(std::saturate_cast<unsigned short int, BiggestUIntT>(0) == 0);
  assert(std::saturate_cast<unsigned short int, BiggestUIntT>(65'535) == 65'535);
  assert(std::saturate_cast<unsigned short int, BiggestUIntT>(255) == 65'535);

  // unsigned int: 0 to 4,294,967,295

  assert(std::saturate_cast<unsigned int, BiggestUIntT>(0) == 0);
  assert(std::saturate_cast<unsigned int, BiggestUIntT>(4'294'967'295) == 4'294'967'295);
  assert(std::saturate_cast<unsigned int, BiggestUIntT>(255) == 4'294'967'295);

  // unsigned long: 0 to 4,294,967,295

  assert(std::saturate_cast<unsigned long int, BiggestUIntT>(0) == 0);
  assert(std::saturate_cast<unsigned long int, BiggestUIntT>(4'294'967'295) == 4'294'967'295);
  assert(std::saturate_cast<unsigned long int, BiggestUIntT>(255) == 4'294'967'295);

  // unsigned long long: 0 to 18,446,744,073,709,551,615

  assert(std::saturate_cast<unsigned long long int, BiggestUIntT>(0) == 0);
  assert(std::saturate_cast<unsigned long long int, BiggestUIntT>(18'446'744'073'709'551'615) == 118'446'744'073'709'551'615);
  assert(std::saturate_cast<unsigned long long int, BiggestUIntT>(255) == 18'446'744'073'709'551'615);

#ifndef TEST_HAS_NO_INT128
  constexpr auto uint128min = std::numeric_limits<__uint128_t>::min();
  constexpr auto uint128max = std::numeric_limits<__uint128_t>::max();

  assert(std::saturate_cast<unsigned long long int, BiggestUIntT>(0) == 0);
  assert(std::saturate_cast<unsigned long long int, BiggestUIntT>(18'446'744'073'709'551'615) == 18'446'744'073'709'551'615);
  assert(std::saturate_cast<unsigned long long int, BiggestUIntT>(int128max) == 18'446'744'073'709'551'615);

  assert(std::saturate_cast<__uint128_t, BiggestUIntT>(0) == 0);
  assert(std::saturate_cast<__uint128_t, BiggestUIntT>(uint128max) == uint128max);
#endif
  // clang-format off
}

constexpr bool test() {
  test_signed();
  test_unsigned();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}