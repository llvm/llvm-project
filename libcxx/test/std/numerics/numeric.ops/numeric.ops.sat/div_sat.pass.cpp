//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <numeric>

// template<class T>
// constexpr T div_sat(T x, T y) noexcept;                     // freestanding

#include <cassert>
#include <concepts>
#include <limits>
#include <numeric>

#include "check_constexpr.h"

template <typename IntegerT>
constexpr bool test_signed() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::div_sat(minVal, maxVal)));

  // No saturation
  {
    std::same_as<IntegerT> decltype(auto) quot = std::div_sat(IntegerT{3}, IntegerT{4});
    assert(quot == IntegerT{0});
  }

  {
    std::same_as<IntegerT> decltype(auto) quot = std::div_sat(maxVal, minVal);
    assert(quot == (maxVal / minVal));
  }

  {
    std::same_as<IntegerT> decltype(auto) quot = std::div_sat(minVal, maxVal);
    assert(quot == (minVal / maxVal));
  }

  // Saturation - max only
  {
    std::same_as<IntegerT> decltype(auto) quot = std::div_sat(minVal, IntegerT{-1});
    assert(quot == maxVal);
  }

  return true;
}

template <typename IntegerT>
constexpr bool test_unsigned() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::div_sat(minVal, maxVal)));

  // No saturation
  {
    std::same_as<IntegerT> decltype(auto) quot = std::div_sat(IntegerT{3}, IntegerT{4});
    assert(quot == IntegerT{0});
  }

  {
    std::same_as<IntegerT> decltype(auto) quot = std::div_sat(minVal, maxVal);
    assert(quot == (minVal / maxVal));
  }

  // Unsigned integer devision never overflow

  return true;
}

template <typename IntegerT>
void test_constexpr() {
  TEST_EXPRESSION_CONSTEXPR(std::div_sat(IntegerT{90}, IntegerT{84}));
  TEST_EXPRESSION_NOT_CONSTEXPR(std::div_sat(IntegerT{90}, IntegerT{0}));
}

constexpr bool test() {
  // Signed
  test_signed<signed char>();
  test_signed<short int>();
  test_signed<int>();
  test_signed<long int>();
  test_signed<long long int>();
#ifndef _LIBCPP_HAS_NO_INT128
  test_signed<__int128_t>();
#endif
  // Unsigned
  test_unsigned<unsigned char>();
  test_unsigned<unsigned short int>();
  test_unsigned<unsigned int>();
  test_unsigned<unsigned long int>();
  test_unsigned<unsigned long long int>();
#ifndef _LIBCPP_HAS_NO_INT128
  test_unsigned<__uint128_t>();
#endif

  return true;
}

bool test_constexpr() {
  // Signed
  test_constexpr<signed char>();
  test_constexpr<short int>();
  test_constexpr<int>();
  test_constexpr<long int>();
  test_constexpr<long long int>();
  // Unsigned
  test_constexpr<unsigned char>();
  test_constexpr<unsigned short int>();
  test_constexpr<unsigned int>();
  test_constexpr<unsigned long int>();
  test_constexpr<unsigned long long int>();

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  assert(test_constexpr());

  return 0;
}
