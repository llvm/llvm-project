//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

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

// std::abs has __int128/_BitInt(N) overloads as a libc++ extension.
// Narrow signed types stay on the abs(int) promotion path.
template <class T>
struct unpromoted_abs : std::is_same<decltype(std::abs(T(0))), T> {};
static_assert(!unpromoted_abs<signed char>::value, "");
static_assert(!unpromoted_abs<short>::value, "");

#if TEST_HAS_BITINT
// Every signed _BitInt(N) gets a same-type abs, including widths narrower than
// int. unsigned _BitInt is excluded by the signed-only contract.
template <class T, class = void>
struct has_abs : std::false_type {};
template <class T>
struct has_abs<T, decltype((void)std::abs(T(0)))> : std::true_type {};
static_assert(has_abs<signed _BitInt(2)>::value, "");
static_assert(has_abs<signed _BitInt(7)>::value, "");
static_assert(has_abs<signed _BitInt(32)>::value, "");
static_assert(has_abs<signed _BitInt(64)>::value, "");
static_assert(!has_abs<unsigned _BitInt(32)>::value, ""); // signed-only contract
static_assert(!has_abs<unsigned _BitInt(64)>::value, "");

template <int N>
void test_signed_bitint() {
  typedef signed _BitInt(N) T;
  ASSERT_SAME_TYPE(decltype(std::abs(T(0))), T);
  assert(std::abs(T(0)) == T(0));
  assert(std::abs(T(1)) == T(1));
  assert(std::abs(T(42)) == T(42));
  assert(std::abs(T(-1)) == T(1));
  assert(std::abs(T(-42)) == T(42));

  // T_MIN omitted: -T_MIN is UB, same as abs(INT_MIN).
  T t_max       = static_cast<T>(~static_cast<unsigned _BitInt(N)>(0) >> 1);
  T t_min_plus1 = -t_max;
  assert(std::abs(t_max) == t_max);
  assert(std::abs(t_min_plus1) == t_max);
}
#endif // TEST_HAS_BITINT

#ifndef TEST_HAS_NO_INT128
void test_int128() {
  ASSERT_SAME_TYPE(decltype(std::abs(static_cast<__int128_t>(0))), __int128_t);
  assert(std::abs(static_cast<__int128_t>(0)) == 0);
  assert(std::abs(static_cast<__int128_t>(42)) == 42);
  assert(std::abs(static_cast<__int128_t>(-42)) == 42);
  assert(std::abs(static_cast<__int128_t>(-1)) == 1);

  // INT128_MIN omitted: -__x is UB, same as abs(INT_MIN).
  __int128_t big_neg = -((static_cast<__int128_t>(1) << 100));
  __int128_t big_pos = static_cast<__int128_t>(1) << 100;
  assert(std::abs(big_neg) == big_pos);

  __int128_t int128_max       = static_cast<__int128_t>(~static_cast<__uint128_t>(0) >> 1);
  __int128_t int128_min_plus1 = -int128_max;
  assert(std::abs(int128_max) == int128_max);
  assert(std::abs(int128_min_plus1) == int128_max);
}
#endif // TEST_HAS_NO_INT128

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

#if TEST_HAS_BITINT
  // Non-byte-aligned _BitInt(N) has uninitialized padding bits; passing such a
  // value to std::abs trips a false-positive MSan report (#204217). Gate those
  // widths out under MSan; byte-aligned widths still run.
  test_signed_bitint<8>();
  test_signed_bitint<16>();
  test_signed_bitint<32>();
  test_signed_bitint<64>();
#  if !TEST_HAS_FEATURE(memory_sanitizer)
  test_signed_bitint<7>();
  test_signed_bitint<33>();
  test_signed_bitint<63>();
  test_signed_bitint<65>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 128
  test_signed_bitint<128>();
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
#    if !TEST_HAS_FEATURE(memory_sanitizer)
  test_signed_bitint<129>();
#    endif
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
#endif // TEST_HAS_BITINT

#ifndef TEST_HAS_NO_INT128
  test_int128();
#endif

  return 0;
}
