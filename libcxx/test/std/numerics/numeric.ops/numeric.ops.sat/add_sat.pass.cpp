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
// constexpr T add_sat(T x, T y) noexcept;                     // freestanding

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
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{3}, IntegerT{4});
    assert(sum == IntegerT{7});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{-3}, IntegerT{4});
    assert(sum == IntegerT{1});
  }

  // Saturation - max - both arguments positive
  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, IntegerT{4});
    assert(sum == maxVal);
  }

  // Saturation - min - both arguments negative
  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, IntegerT{-4});
    assert(sum == minVal);
  }

  return true;
}

template <typename IntegerT>
constexpr bool test_unsigned() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::div_sat(minVal, maxVal)));

  // No Saturation
  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{3}, IntegerT{4});
    assert(sum == IntegerT{7});
  }

  // Saturation - max only
  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, IntegerT{4});
    assert(sum == maxVal);
  }

  return true;
}

constexpr bool test() {
  // signed
  test_signed<signed char>();
  test_signed<short int>();
  test_signed<int>();
  test_signed<long int>();
  test_signed<long long int>();
  // unsigned
  test_unsigned<unsigned char>();
  test_unsigned<unsigned short int>();
  test_unsigned<unsigned int>();
  test_unsigned<unsigned long int>();
  test_unsigned<unsigned long long int>();

  return true;
}

// ADDITIONAL_COMPILE_FLAGS: -Wno-constant-conversion

constexpr void cppreference_test() {
  {
    constexpr int a = std::add_sat(3, 4); // no saturation occurs, T = int
    static_assert(a == 7);

    constexpr unsigned char b = std::add_sat<unsigned char>(UCHAR_MAX, 4); // saturated
    static_assert(b == UCHAR_MAX);

    constexpr unsigned char c = std::add_sat(UCHAR_MAX, 4); // not saturated, T = int
                                                            // add_sat(int, int) returns int tmp == 259,
                                                            // then assignment truncates 259 % 256 == 3
    static_assert(c == 3);

    //  unsigned char d = std::add_sat(252, c); // Error: inconsistent deductions for T

    constexpr unsigned char e = std::add_sat<unsigned char>(251, a); // saturated
    static_assert(e == UCHAR_MAX);
    // 251 is of type T = unsigned char, `a` is converted to unsigned char value;
    // might yield an int -> unsigned char conversion warning for `a`

    constexpr signed char f = std::add_sat<signed char>(-123, -3); // not saturated
    static_assert(f == -126);

    constexpr signed char g = std::add_sat<signed char>(-123, -13); // saturated
    static_assert(g == std::numeric_limits<signed char>::min());    // g == -128
  }
}

int main(int, char**) {
  test();
  static_assert(test());
  cppreference_test();

  return 0;
}
