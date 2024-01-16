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
#include <limits>

template<typename T, typename U>
concept can_sub_sat
  = requires(T t, U u) { { std::sub_sat(t, u) } -> std::same_as<T>; };

static_assert( can_sub_sat<int, int> );
static_assert( not can_sub_sat<int, short> );
static_assert( not can_sub_sat<unsigned, int> );
static_assert( noexcept(std::sub_sat(0, 0)) );

using std::sub_sat;

// Signed type
static_assert(sub_sat(0, 0) == 0);
static_assert(sub_sat(1, 1) == 0);
static_assert(sub_sat(-1, -1) == 0);
static_assert(sub_sat(-1, 1) == -2);
constexpr auto max = std::numeric_limits<int>::max();
constexpr auto min = std::numeric_limits<int>::min();
static_assert(sub_sat(max, 1) == max - 1);
static_assert(sub_sat(1, max) == 1 - max);
static_assert(sub_sat(max, max) == 0);
static_assert(sub_sat(min, 1) == min);
static_assert(sub_sat(min, 123) == min);
static_assert(sub_sat(0, max) == min + 1);
static_assert(sub_sat(-1, max) == min);
static_assert(sub_sat(-2, max) == min);
static_assert(sub_sat(-2, min) == max - 1);
static_assert(sub_sat(-1, min) == max);
static_assert(sub_sat(0, min) == max);
static_assert(sub_sat(1, min) == max);
static_assert(sub_sat(min, -1) == min + 1);
static_assert(sub_sat(min, min) == 0);
static_assert(sub_sat(max, min) == max);
static_assert(sub_sat(min, max) == min);

// Wider signed type than the args
static_assert(sub_sat<long long>(max, min) == (long long)max * 2 + 1);
static_assert(sub_sat<long long>(min, max) == (long long)min * 2 + 1);

// Signed type that undergoes integer promotion
constexpr auto shrt_max = std::numeric_limits<short>::max();
constexpr auto shrt_min = std::numeric_limits<short>::min();
static_assert(sub_sat<short>(0, 0) == 0);
static_assert(sub_sat<short>(1, 1) == 0);
static_assert(sub_sat<short>(3, 1) == 2);
static_assert(sub_sat<short>(shrt_max, shrt_max) == 0);
static_assert(sub_sat<short>(shrt_max, 1) == shrt_max - 1);
static_assert(sub_sat<short>(1, shrt_max) == shrt_min + 2);
static_assert(sub_sat<short>(shrt_max, shrt_min) == shrt_max);
static_assert(sub_sat<short>(0, shrt_min) == shrt_max);
static_assert(sub_sat<short>(shrt_min, (short)1) == shrt_min);
static_assert(sub_sat<short>(shrt_min, (short)-1) == shrt_min + 1);
static_assert(sub_sat<short>((short)-1, shrt_min) == shrt_max);
static_assert(sub_sat<short>((short)1, shrt_min) == shrt_max);

// Unsigned type
static_assert(sub_sat(0u, 0u) == 0u);
static_assert(sub_sat(1u, 1u) == 0u);
static_assert(sub_sat(-1u, -1u) == 0u);
static_assert(sub_sat(-1u, 1u) == -2u);
constexpr auto umax = std::numeric_limits<unsigned>::max();
static_assert(sub_sat(0u, 1u) == 0u);
static_assert(sub_sat(umax, umax) == 0u);
static_assert(sub_sat(umax, 0u) == umax);
static_assert(sub_sat(0u, umax) == 0u);
static_assert(sub_sat(umax, 1u) == umax - 1u);
static_assert(sub_sat(0u, 0u) == 0u);

// Wider unsigned type than the args
static_assert(sub_sat<unsigned long long>(0u, umax) == 0u);

// Unsigned type that undergoes integer promotion
constexpr auto ushrt_max = std::numeric_limits<unsigned short>::max();
static_assert(sub_sat<unsigned short>(0, 0) == 0);
static_assert(sub_sat<unsigned short>(1, 1) == 0);
static_assert(sub_sat<unsigned short>(3, 1) == 2);
static_assert(sub_sat<unsigned short>(ushrt_max, ushrt_max) == 0);
static_assert(sub_sat<unsigned short>(0, 1) == 0);
static_assert(sub_sat<unsigned short>(1, ushrt_max) == 0);
