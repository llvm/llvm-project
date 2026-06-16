//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "test_macros.h"

template <class T>
struct correct_size_int {
  typedef typename std::conditional<sizeof(T) < sizeof(int), int, T>::type type;
};

template <class Source, class Result>
void test_abs() {
  Source neg_val = -5;
  Source pos_val = 5;
  Result res     = 5;

  ASSERT_SAME_TYPE(decltype(std::abs(neg_val)), Result);

  assert(std::abs(neg_val) == res);
  assert(std::abs(pos_val) == res);
}

void test_big() {
  long long int big_value          = std::numeric_limits<long long int>::max(); // a value too big for ints to store
  long long int negative_big_value = -big_value;
  assert(std::abs(negative_big_value) == big_value); // make sure it doesn't get casted to a smaller type
}

// The libc++ extension overload covers signed integer types >= sizeof(int)
// that aren't matched by the standard abs(int/long/long long): __int128 and
// signed _BitInt(N). Shorter types still go through abs(int) via integer
// promotion -- the new template's sizeof gate must keep them out.
template <class T>
constexpr bool unpromoted_abs = std::is_same<decltype(std::abs(T(0))), T>::value;
static_assert(!unpromoted_abs<signed char>);
static_assert(!unpromoted_abs<short>);

#ifdef __BITINT_MAXWIDTH__
// _BitInt(N) does not participate in integer promotion. The template's
// sizeof(_Tp) >= sizeof(int) gate determines membership, not bit width:
// sizeof(_BitInt(31)) == sizeof(int) on x86_64 so the overload includes it.
// Probe via SFINAE on widths whose sizeof is unambiguous.
template <class T, class = void>
constexpr bool has_abs = false;
template <class T>
constexpr bool has_abs<T, decltype((void)std::abs(T(0)))> = true;
static_assert(!has_abs<signed _BitInt(7)>);
static_assert(has_abs<signed _BitInt(32)>);
static_assert(has_abs<signed _BitInt(64)>);
// Pin the signed-only contract of __is_signed_integer_v: a regression that
// accepted unsigned _BitInt would otherwise slip past every other test here.
static_assert(!has_abs<unsigned _BitInt(32)>);
static_assert(!has_abs<unsigned _BitInt(64)>);

template <int N>
void test_signed_bitint() {
  using T = signed _BitInt(N);
  ASSERT_SAME_TYPE(decltype(std::abs(T(0))), T);
  assert(std::abs(T(0)) == T(0));
  assert(std::abs(T(1)) == T(1));
  assert(std::abs(T(42)) == T(42));
  assert(std::abs(T(-1)) == T(1));
  assert(std::abs(T(-42)) == T(42));

  // T_MAX has no overflow (identity). T_MIN + 1 negates cleanly to T_MAX.
  // T_MIN itself is intentionally not tested -- -T_MIN overflows, same caveat
  // as std::abs(int) with INT_MIN.
  T t_max       = static_cast<T>(~static_cast<unsigned _BitInt(N)>(0) >> 1);
  T t_min_plus1 = -t_max;
  assert(std::abs(t_max) == t_max);
  assert(std::abs(t_min_plus1) == t_max);
}
#endif

#ifndef TEST_HAS_NO_INT128
void test_int128() {
  ASSERT_SAME_TYPE(decltype(std::abs(static_cast<__int128_t>(0))), __int128_t);
  assert(std::abs(static_cast<__int128_t>(0)) == 0);
  assert(std::abs(static_cast<__int128_t>(42)) == 42);
  assert(std::abs(static_cast<__int128_t>(-42)) == 42);
  assert(std::abs(static_cast<__int128_t>(-1)) == 1);

  // INT128_MIN is unrepresentable as a positive __int128. Skip, consistent
  // with the standard abs(int) INT_MIN caveat.
  __int128_t big_neg = -((static_cast<__int128_t>(1) << 100));
  __int128_t big_pos = static_cast<__int128_t>(1) << 100;
  assert(std::abs(big_neg) == big_pos);

  __int128_t int128_max       = static_cast<__int128_t>(~static_cast<__uint128_t>(0) >> 1);
  __int128_t int128_min_plus1 = -int128_max;
  assert(std::abs(int128_max) == int128_max);
  assert(std::abs(int128_min_plus1) == int128_max);
}
#endif

// The following is helpful to keep in mind:
// 1byte == char <= short <= int <= long <= long long

int main(int, char**) {
  // On some systems char is unsigned.
  // If that is the case, we should just test signed char twice.
  typedef std::conditional< std::is_signed<char>::value, char, signed char >::type SignedChar;

  // All types less than or equal to and not greater than int are promoted to int.
  test_abs<short int, int>();
  test_abs<SignedChar, int>();
  test_abs<signed char, int>();

  // These three calls have specific overloads:
  test_abs<int, int>();
  test_abs<long int, long int>();
  test_abs<long long int, long long int>();

  // Here there is no guarantee that int is larger than int8_t so we
  // use a helper type trait to conditional test against int.
  test_abs<std::int8_t, correct_size_int<std::int8_t>::type>();
  test_abs<std::int16_t, correct_size_int<std::int16_t>::type>();
  test_abs<std::int32_t, correct_size_int<std::int32_t>::type>();
  test_abs<std::int64_t, correct_size_int<std::int64_t>::type>();

  test_abs<long double, long double>();
  test_abs<double, double>();
  test_abs<float, float>();

  test_big();

#ifdef __BITINT_MAXWIDTH__
  // _BitInt(N < sizeof(int) * CHAR_BIT) does not match the new template
  // (sizeof guard) and does not promote, so abs has no overload. Start at 32.
  test_signed_bitint<32>();
  test_signed_bitint<64>();
  test_signed_bitint<33>();
  test_signed_bitint<63>();
  test_signed_bitint<65>();
#  if __BITINT_MAXWIDTH__ >= 128
  test_signed_bitint<128>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
  test_signed_bitint<129>();
  test_signed_bitint<256>();

  // Large value: |-2^200| == 2^200.
  signed _BitInt(256) v        = -(static_cast<signed _BitInt(256)>(1) << 200);
  signed _BitInt(256) expected = static_cast<signed _BitInt(256)>(1) << 200;
  assert(std::abs(v) == expected);
#  endif
#  if __BITINT_MAXWIDTH__ >= 1024
  test_signed_bitint<512>();
  test_signed_bitint<1024>();
#  endif
#endif

#ifndef TEST_HAS_NO_INT128
  test_int128();
#endif

  return 0;
}
