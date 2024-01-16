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
// constexpr T mul_sat(T x, T y) noexcept;                     // freestanding

#include <cassert>
#include <concepts>
#include <limits>
#include <numeric>

template <typename IntegerT>
constexpr bool test_signed() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::mul_sat(minVal, maxVal)));

  // No saturation
  {
    std::same_as<IntegerT> decltype(auto) prod = std::mul_sat(IntegerT{3}, IntegerT{4});
    assert(prod == IntegerT{12});
  }

  // Saturation - max - both arguments positive
  {
    std::same_as<IntegerT> decltype(auto) prod = std::mul_sat(maxVal, IntegerT{4});
    assert(prod == maxVal);
  }

  // Saturation - max - both arguments negative
  {
    std::same_as<IntegerT> decltype(auto) prod = std::mul_sat(minVal, IntegerT{-4});
    assert(prod == maxVal);
  }

  // Saturation - min - left positive, right negative
  {
    std::same_as<IntegerT> decltype(auto) prod = std::mul_sat(maxVal, IntegerT{-4});
    assert(prod == minVal);
  }

  // Saturation - min - left negative, right positive
  {
    std::same_as<IntegerT> decltype(auto) prod = std::mul_sat(minVal, IntegerT{4});
    assert(prod == minVal);
  }

  return true;
}

template <typename IntegerT>
constexpr bool test_unsigned() {
  constexpr auto minVal = std::numeric_limits<IntegerT>::min();
  constexpr auto maxVal = std::numeric_limits<IntegerT>::max();

  static_assert(noexcept(std::mul_sat(minVal, maxVal)));

  // No saturation
  {
    std::same_as<IntegerT> decltype(auto) prod = std::mul_sat(IntegerT{3}, IntegerT{4});
    assert(prod == IntegerT{12});
  }

  // Saturation
  {
    std::same_as<IntegerT> decltype(auto) prod = std::mul_sat(maxVal, IntegerT{4});
    assert(prod == maxVal);
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

#include <numeric>
#include <climits>

template<typename T, typename U>
concept can_mul_sat
  = requires(T t, U u) { { std::mul_sat(t, u) } -> std::same_as<T>; };

static_assert( can_mul_sat<int, int> );
static_assert( not can_mul_sat<int, short> );
static_assert( not can_mul_sat<unsigned, int> );
static_assert( noexcept(std::mul_sat(0, 0)) );

using std::mul_sat;

static_assert(mul_sat(1, 1) == 1);
static_assert(mul_sat(10, 11) == 110);
static_assert(mul_sat(INT_MAX / 2, 3) == INT_MAX);
static_assert(mul_sat(INT_MAX / 2, -3) == INT_MIN);
static_assert(mul_sat(INT_MAX / -2, 3) == INT_MIN);
static_assert(mul_sat(INT_MIN / 2, -3) == INT_MAX);
static_assert(mul_sat(INT_MIN, -1) == INT_MAX);
static_assert(mul_sat(INT_MAX, -1) == INT_MIN + 1);
static_assert(mul_sat(INT_MAX, INT_MAX) == INT_MAX);
static_assert(mul_sat(INT_MAX, -INT_MAX) == INT_MIN);
static_assert(mul_sat(UINT_MAX, UINT_MAX) == UINT_MAX);
static_assert(mul_sat(UINT_MAX, 0u) == 0);
static_assert(mul_sat(0u, UINT_MAX) == 0);
static_assert(mul_sat((short)SHRT_MAX, (short)2) == SHRT_MAX);
static_assert(mul_sat((short)SHRT_MAX, (short)SHRT_MIN) == SHRT_MIN);
static_assert(mul_sat<long long>(SHRT_MAX, 2) == 2L * SHRT_MAX);
