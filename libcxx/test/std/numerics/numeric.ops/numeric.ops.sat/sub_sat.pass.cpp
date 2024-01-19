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
// constexpr T sub_sat(T x, T y) noexcept;                     // freestanding

#include <cassert>
#include <concepts>
#include <limits>
#include <numeric>

template <typename IntegerT>
constexpr bool test_signed() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::sub_sat(minVal, maxVal)));

  // No saturation
  {
    std::same_as<IntegerT> decltype(auto) diff = std::sub_sat(IntegerT{3}, IntegerT{4});
    assert(diff == IntegerT{-1});
  }

  // Saturation - min - left negative, right positive
  {
    std::same_as<IntegerT> decltype(auto) diff = std::sub_sat(minVal, IntegerT{4});
    assert(diff == minVal);
  }

  {
    constexpr IntegerT x = minVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = maxVal / IntegerT{2} + IntegerT{28};

    std::same_as<IntegerT> decltype(auto) diff = std::sub_sat(x, y);
    assert(diff == minVal);
  }

  // Saturation - max - left postitive, right negative
  {
    std::same_as<IntegerT> decltype(auto) diff = std::sub_sat(maxVal, IntegerT{-4});
    assert(diff == maxVal);
  }

  {
    constexpr IntegerT x = maxVal / IntegerT{2} + IntegerT{28};
    constexpr IntegerT y = minVal / IntegerT{2} + IntegerT{27};

    std::same_as<IntegerT> decltype(auto) diff = std::sub_sat(x, y);
    assert(diff == maxVal);
  }

  return true;
}

template <typename IntegerT>
constexpr bool test_unsigned() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::sub_sat(minVal, maxVal)));

  // No saturation
  {
    std::same_as<IntegerT> decltype(auto) diff = std::sub_sat(IntegerT{3}, IntegerT{1});
    assert(diff == IntegerT{2});
  }

  // Saturation - min only
  {
    std::same_as<IntegerT> decltype(auto) diff = std::sub_sat(minVal, IntegerT{4});
    assert(diff == minVal);
  }

  {
    constexpr IntegerT x = minVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = maxVal / IntegerT{2} + IntegerT{28};

    std::same_as<IntegerT> decltype(auto) diff = std::sub_sat(x, y);
    assert(diff == minVal);
  }

  return true;
}

constexpr bool test() {
  // Signed
  test_signed<signed char>();
  test_signed<short int>();
  test_signed<int>();
  test_signed<long int>();
  test_signed<long long int>();
#ifndef TEST_HAS_NO_INT128
  test_signed<__int128_t>();
#endif
  // Unsigned
  test_unsigned<unsigned char>();
  test_unsigned<unsigned short int>();
  test_unsigned<unsigned int>();
  test_unsigned<unsigned long int>();
  test_unsigned<unsigned long long int>();
#ifndef TEST_HAS_NO_INT128
  test_unsigned<__uint128_t>();
#endif

  return true;
}

int main(int, char**) {
  assert();
  static_assert(test());

  return 0;
}
