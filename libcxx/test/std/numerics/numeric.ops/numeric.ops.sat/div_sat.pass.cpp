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

template <typename IntegerT>
constexpr bool test_signed() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::div_sat(minVal, maxVal)));

  // No saturation
  {
    std::same_as<IntegerT> decltype(auto) div = std::div_sat(IntegerT{3}, IntegerT{4});
    assert(div == IntegerT{0});
  }

  {
    std::same_as<IntegerT> decltype(auto) div = std::div_sat(maxVal, minVal);
    assert(div == (maxVal / minVal));
  }

  {
    std::same_as<IntegerT> decltype(auto) div = std::div_sat(minVal, maxVal);
    assert(div == (minVal / maxVal));
  }

  // Saturation - max only
  {
    std::same_as<IntegerT> decltype(auto) div = std::div_sat(minVal, IntegerT{-1});
    assert(div == maxVal);
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
    std::same_as<IntegerT> decltype(auto) div = std::div_sat(IntegerT{3}, IntegerT{4});
    assert(div == IntegerT{0});
  }

  {
    std::same_as<IntegerT> decltype(auto) div = std::div_sat(minVal, maxVal);
    assert(div == (minVal / maxVal));
  }

  // Unsigned integer devision never overflow

  return true;
}

constexpr bool test() {
  // Signed
  test_signed<signed char>();
  test_signed<short int>();
  test_signed<int>();
  test_signed<long int>();
  test_signed<long long int>();
  // Unsigned
  test_unsigned<unsigned char>();
  test_unsigned<unsigned short int>();
  test_unsigned<unsigned int>();
  test_unsigned<unsigned long int>();
  test_unsigned<unsigned long long int>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
