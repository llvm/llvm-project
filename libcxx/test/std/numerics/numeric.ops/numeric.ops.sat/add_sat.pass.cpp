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

  // No saturation (-1, 0, 1)
  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{0}, IntegerT{0});
    assert(sum == IntegerT{0});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{0}, IntegerT{1});
    assert(sum == IntegerT{1});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{1}, IntegerT{0});
    assert(sum == IntegerT{1});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{0}, IntegerT{-1});
    assert(sum == IntegerT{-1});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{-1}, IntegerT{0});
    assert(sum == IntegerT{-1});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{1}, IntegerT{1});
    assert(sum == IntegerT{2});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{1}, IntegerT{-1});
    assert(sum == IntegerT{0});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{-1}, IntegerT{1});
    assert(sum == IntegerT{0});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{-1}, IntegerT{-1});
    assert(sum == IntegerT{-2});
  }

  // No saturation (any value)

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{27}, IntegerT{28});
    assert(sum == IntegerT{55});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(IntegerT{-27}, IntegerT{28});
    assert(sum == IntegerT{1});
  }

  // No saturation (min, -1, 0, 1, max)

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, IntegerT{0});
    assert(sum == minVal);
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, IntegerT{1});
    assert(sum == minVal + IntegerT{1});
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, IntegerT{0});
    assert(sum == maxVal);
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
    // Large values
    constexpr IntegerT x = maxVal / IntegerT{2} + IntegerT{27};
    constexpr IntegerT y = maxVal / IntegerT{2} + IntegerT{28};

    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(x, y);
    assert(sum == maxVal);
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, maxVal);
    assert(sum == maxVal);
  }

  // Saturation - min - both arguments negative
  {
    // Large values
    constexpr IntegerT x = minVal / IntegerT{2} + IntegerT{-27};
    constexpr IntegerT y = minVal / IntegerT{2} + IntegerT{-28};

    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(x, y);
    assert(sum == minVal);
  }

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, minVal);
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

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(minVal, minVal);
    assert(sum == minVal);
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

  {
    std::same_as<IntegerT> decltype(auto) sum = std::add_sat(maxVal, maxVal);
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


#include <numeric>
#include <limits>

template<typename T, typename U>
concept can_add_sat
  = requires(T t, U u) { { std::add_sat(t, u) } -> std::same_as<T>; };

static_assert( can_add_sat<int, int> );
static_assert( not can_add_sat<int, short> );
static_assert( not can_add_sat<unsigned, int> );
static_assert( noexcept(std::add_sat(0, 0)) );

using std::add_sat;

// Signed type
static_assert(add_sat(0, 0) == 0);
static_assert(add_sat(1, 1) == 2);
static_assert(add_sat(-1, -1) == -2);
static_assert(add_sat(-1, 1) == 0);
constexpr auto max = std::numeric_limits<int>::max();
constexpr auto min = std::numeric_limits<int>::min();
static_assert(add_sat(max, 1) == max);
static_assert(add_sat(1, max) == max);
static_assert(add_sat(max, max) == max);
static_assert(add_sat(min, -1) == min);
static_assert(add_sat(-1, min) == min);
static_assert(add_sat(min, min) == min);
static_assert(add_sat(max, min) == -1);
static_assert(add_sat(min, max) == -1);

// Wider signed type than the args
static_assert(add_sat<long long>(max, max) == (long long)max * 2);
static_assert(add_sat<long long>(min, min) == (long long)min * 2);

// Signed type that undergoes integer promotion
constexpr auto shrt_max = std::numeric_limits<short>::max();
constexpr auto shrt_min = std::numeric_limits<short>::min();
static_assert(add_sat<short>(0, 0) == 0);
static_assert(add_sat<short>(1, 1) == 2);
static_assert(add_sat<short>(shrt_max, shrt_max) == shrt_max);
static_assert(add_sat<short>(shrt_max, 1) == shrt_max);
static_assert(add_sat<short>(1, shrt_max) == shrt_max);
static_assert(add_sat<short>(shrt_min, (short)-1) == shrt_min);
static_assert(add_sat<short>((short)-1, shrt_min) == shrt_min);
static_assert(add_sat<short>(shrt_min, (short)1) == -shrt_max);
static_assert(add_sat<short>((short)1, shrt_min) == -shrt_max);

// Unsigned type
static_assert(add_sat(0u, 0u) == 0u);
static_assert(add_sat(1u, 1u) == 2u);
constexpr auto umax = std::numeric_limits<unsigned>::max();
static_assert(add_sat(umax, 1u) == umax);
static_assert(add_sat(1u, umax) == umax);
static_assert(add_sat(umax, umax) == umax);
static_assert(add_sat(0u, umax) == umax);
static_assert(add_sat(umax, 0u) == umax);
static_assert(add_sat(0u, 1u) == 1u);
static_assert(add_sat(1u, 0u) == 1u);

// Wider unsigned type than the args
static_assert(add_sat<unsigned long long>(umax, umax) == (long long)umax * 2);

// Unsigned type that undergoes integer promotion
constexpr auto ushrt_max = std::numeric_limits<unsigned short>::max();
static_assert(add_sat<unsigned short>(0, 0) == 0);
static_assert(add_sat<unsigned short>(1, 1) == 2);
static_assert(add_sat<unsigned short>(ushrt_max, ushrt_max) == ushrt_max);
static_assert(add_sat<unsigned short>(ushrt_max, 1) == ushrt_max);
static_assert(add_sat<unsigned short>(1, ushrt_max) == ushrt_max);
