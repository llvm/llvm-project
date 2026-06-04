//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::abs for _BitInt(N) and __int128 -- libc++ extension over the standard
// abs(int / long / long long) overloads. The new template covers signed
// integer types not handled by the existing builtin overloads, gated on
// __is_signed_integer_v plus a sizeof(_Tp) >= sizeof(int) check that keeps
// shorter standard types on the integer-promotion path.

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cassert>
#include <cstdlib>
#include <type_traits>

#include "test_macros.h"

// Pin down the sizeof gate: narrow signed types still resolve to abs(int) via
// integer promotion, and the new template stays out of their overload set.
// Detection probes the template path specifically (the type of std::abs(t) is
// the unpromoted T only for the new overload).
template <class T>
constexpr bool unpromoted_abs = std::is_same<decltype(std::abs(T(0))), T>::value;
static_assert(!unpromoted_abs<signed char>);
static_assert(!unpromoted_abs<short>);

#if TEST_HAS_EXTENSION(bit_int)
// _BitInt(N) narrower than int does not promote and the new template's sizeof
// gate excludes it, so abs is not callable. Probe via SFINAE.
template <class T, class = void>
constexpr bool has_abs = false;
template <class T>
constexpr bool has_abs<T, decltype((void)std::abs(T(0)))> = true;
static_assert(!has_abs<signed _BitInt(7)>);
static_assert(!has_abs<signed _BitInt(16)>);
static_assert(!has_abs<signed _BitInt(31)>);
static_assert(has_abs<signed _BitInt(32)>);

template <int N>
void test_signed_bitint() {
  using T = signed _BitInt(N);
  ASSERT_SAME_TYPE(decltype(std::abs(T(0))), T);
  assert(std::abs(T(0)) == T(0));
  assert(std::abs(T(1)) == T(1));
  assert(std::abs(T(42)) == T(42));
  assert(std::abs(T(-1)) == T(1));
  assert(std::abs(T(-42)) == T(42));

  // Boundary cases. T_MAX has no overflow (identity). T_MIN + 1 negates to
  // T_MAX, which is the largest negative value abs handles cleanly. T_MIN
  // itself is intentionally not tested: -T_MIN overflows in signed
  // arithmetic, which is undefined behaviour. The same caveat applies to
  // std::abs(int) with INT_MIN.
  T t_max       = static_cast<T>(~static_cast<unsigned _BitInt(N)>(0) >> 1);
  T t_min_plus1 = -t_max; // == T_MIN + 1
  assert(std::abs(t_max) == t_max);
  assert(std::abs(t_min_plus1) == t_max);
}
#endif

int main(int, char**) {
#if TEST_HAS_EXTENSION(bit_int)
  // _BitInt(N) with N < sizeof(int) * CHAR_BIT does not match the new
  // template (sizeof guard) and does not implicit-promote to int either, so
  // it has no abs overload. Start at 32 bits.
  test_signed_bitint<32>();
  test_signed_bitint<64>();

  // Odd widths >= 32 bits.
  test_signed_bitint<33>();
  test_signed_bitint<63>();
  test_signed_bitint<65>();

#  if __BITINT_MAXWIDTH__ >= 128
  test_signed_bitint<128>();
#  endif

#  if __BITINT_MAXWIDTH__ >= 256
  test_signed_bitint<129>();
  test_signed_bitint<256>();

  // Large value: |-2^200| == 2^200. Python: abs(-(1 << 200)) == 1 << 200.
  signed _BitInt(256) v        = -(static_cast<signed _BitInt(256)>(1) << 200);
  signed _BitInt(256) expected = static_cast<signed _BitInt(256)>(1) << 200;
  assert(std::abs(v) == expected);
#  endif

#  if __BITINT_MAXWIDTH__ >= 1024
  test_signed_bitint<512>();
  test_signed_bitint<1024>();
#  endif
#endif // TEST_HAS_EXTENSION(bit_int)

#if _LIBCPP_HAS_INT128
  ASSERT_SAME_TYPE(decltype(std::abs(static_cast<__int128_t>(0))), __int128_t);
  assert(std::abs(static_cast<__int128_t>(0)) == 0);
  assert(std::abs(static_cast<__int128_t>(42)) == 42);
  assert(std::abs(static_cast<__int128_t>(-42)) == 42);
  assert(std::abs(static_cast<__int128_t>(-1)) == 1);

  // INT128_MIN is unrepresentable as a positive __int128; -__x overflows.
  // Skip that case -- the standard's abs(int) has the same issue with
  // INT_MIN, so this is consistent.
  __int128_t big_neg = -((static_cast<__int128_t>(1) << 100));
  __int128_t big_pos = static_cast<__int128_t>(1) << 100;
  assert(std::abs(big_neg) == big_pos);

  // Boundary: INT128_MAX (identity) and INT128_MIN+1 (negation cleanly
  // produces INT128_MAX).
  __int128_t int128_max       = static_cast<__int128_t>(~static_cast<__uint128_t>(0) >> 1);
  __int128_t int128_min_plus1 = -int128_max;
  assert(std::abs(int128_max) == int128_max);
  assert(std::abs(int128_min_plus1) == int128_max);
#endif

  return 0;
}
