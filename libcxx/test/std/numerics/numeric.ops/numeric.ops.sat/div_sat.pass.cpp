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

#include <numeric>
#include <climits>

template<typename T, typename U>
concept can_div_sat
  = requires(T t, U u) { { std::div_sat(t, u) } -> std::same_as<T>; };

static_assert( can_div_sat<int, int> );
static_assert( not can_div_sat<int, short> );
static_assert( not can_div_sat<unsigned, int> );
static_assert( noexcept(std::div_sat(0, 1)) );

using std::div_sat;

static_assert(std::div_sat(0, 1) == 0);
static_assert(std::div_sat(0, -1) == 0);
static_assert(std::div_sat(1, -1) == -1);
static_assert(std::div_sat(10, -2) == -5);
static_assert(std::div_sat(-10, -2) == 5);
static_assert(std::div_sat(INT_MAX, 1) == INT_MAX);
static_assert(std::div_sat(INT_MIN, 1) == INT_MIN);
static_assert(std::div_sat(INT_MIN + 1, -1) == INT_MAX);
static_assert(std::div_sat(0u, 1u) == 0u);
static_assert(std::div_sat(UINT_MAX, 1u) == UINT_MAX);
static_assert(std::div_sat(INT_MIN, -1) == INT_MAX);
static_assert(std::div_sat((short)SHRT_MIN, (short)-1) == SHRT_MAX);
static_assert(std::div_sat(LONG_MIN, -1L) == LONG_MAX);
static_assert(std::div_sat(LLONG_MIN, -1LL) == LLONG_MAX);

template<auto N>
std::integral_constant<decltype(N), std::div_sat(N, N-N)>
div_sat_by_zero();

template<auto N>
concept can_div_sat_by_zero = requires { div_sat_by_zero<N>(); };

static_assert( not can_div_sat_by_zero<0> );
static_assert( not can_div_sat_by_zero<1> );
static_assert( not can_div_sat_by_zero<1u> );
static_assert( not can_div_sat_by_zero<-1L> );
static_assert( not can_div_sat_by_zero<short(99)> );
