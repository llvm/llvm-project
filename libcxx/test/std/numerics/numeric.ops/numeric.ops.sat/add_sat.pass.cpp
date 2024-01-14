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
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{27}, IntegerT{28});
    assert(sum == IntegerT{55});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{-27}, IntegerT{28});
    assert(sum == IntegerT{1});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, IntegerT{0});
    assert(sum == minVal);
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, IntegerT{0});
    assert(sum == maxVal);
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, IntegerT{1});
    assert(sum == minVal + IntegerT{1});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, IntegerT{-1});
    assert(sum == maxVal + IntegerT{-1});
  }

  // Saturation - max - both arguments positive
  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, IntegerT{27});
    assert(sum == maxVal);
  }

  {
    constexpr IntegerT x = maxVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = maxVal / IntegerT{2} + IntegerT{28};

    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(x, y);
    assert(sum == maxVal);
  }

  // Saturation - min - both arguments negative
  {
    constexpr IntegerT x = minVal / IntegerT{2} + IntegerT{-27};
    constexpr IntegerT y = minVal / IntegerT{2} + IntegerT{-28};

    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(x, y);
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
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{27}, IntegerT{28});
    assert(sum == IntegerT{55});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, IntegerT{0});
    assert(sum == IntegerT{0});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, IntegerT{1});
    assert(sum == IntegerT{1});
  }

  // Saturation - max only
  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, IntegerT{27});
    assert(sum == maxVal);
  }

  {
    constexpr IntegerT x = maxVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = maxVal / IntegerT{2} + IntegerT{28};

    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(x, y);
    assert(sum == maxVal);
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

int main(int, char**) {
  assert(test());
  static_assert(test());

  return 0;
}
