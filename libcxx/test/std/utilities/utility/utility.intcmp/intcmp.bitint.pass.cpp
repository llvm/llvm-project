//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <utility>

// cmp_equal, cmp_not_equal, cmp_less, cmp_less_equal, cmp_greater,
// cmp_greater_equal, in_range applied to _BitInt(N).
//
// Exercises the four implementation branches of cmp_less/cmp_equal:
//   1. same-signedness shortcut                             (__t < __u)
//   2. both promote to int                                  (branch via int)
//   3. both promote to long long                            (branch via long long)
//   4. fallback using make_unsigned_t                       (wider than long long)
//
// _BitInt widths chosen to land in each branch:
//   -  _BitInt(7)   sizeof==1, < sizeof(int)                -> branch 2
//   -  _BitInt(13)  sizeof==2                               -> branch 2
//   -  _BitInt(32)  sizeof==4 == sizeof(int)                -> branch 2 (signed) / 3 (unsigned)
//   -  _BitInt(33)  sizeof==8                               -> branch 3
//   -  _BitInt(63)  sizeof==8                               -> branch 3
//   -  _BitInt(65)  sizeof==16                              -> branch 4
//   -  _BitInt(128) sizeof==16                              -> branch 4
//   -  _BitInt(200) sizeof==32 (requires __BITINT_MAXWIDTH__ >= 200)
//                                                           -> branch 4

#include <cassert>
#include <limits>
#include <utility>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int)

template <class T, class U>
constexpr bool test_same_sign() {
  // Branch 1: same signedness. Trivial equality/ordering.
  static_assert(std::cmp_equal(T(0), U(0)));
  static_assert(std::cmp_equal(T(42), U(42)));
  static_assert(!std::cmp_equal(T(0), U(1)));
  static_assert(std::cmp_less(T(0), U(1)));
  static_assert(!std::cmp_less(T(1), U(0)));
  static_assert(std::cmp_less_equal(T(1), U(1)));
  static_assert(std::cmp_greater_equal(T(1), U(1)));
  static_assert(std::cmp_not_equal(T(0), U(1)));
  return true;
}

template <class S, class U>
constexpr bool test_mixed_sign() {
  // Signed vs unsigned of the SAME width: negative signed values must
  // compare less than any unsigned value, regardless of the promotion
  // branch chosen.
  constexpr auto s_min = std::numeric_limits<S>::min();
  constexpr auto u_max = std::numeric_limits<U>::max();

  static_assert(std::cmp_less(S(-1), U(0)));
  static_assert(!std::cmp_equal(S(-1), U(-1))); // U(-1) wraps to u_max
  static_assert(std::cmp_less(s_min, U(0)));
  static_assert(std::cmp_greater(u_max, S(0)));
  static_assert(std::cmp_greater(u_max, s_min));
  static_assert(std::cmp_less_equal(S(-1), U(0)));
  static_assert(std::cmp_greater_equal(U(0), S(-1)));

  // Equal-value mixed-sign: a non-negative signed value must compare
  // equal to the corresponding unsigned value.
  static_assert(std::cmp_equal(S(7), U(7)));
  static_assert(std::cmp_equal(U(7), S(7)));
  return true;
}

template <class S, class U>
constexpr bool test_in_range() {
  // in_range relies on numeric_limits<_Tp>::min/max, which requires
  // the digits10 fix (#193002) to be correct for odd _BitInt widths.

  // Signed target: value in range.
  static_assert(std::in_range<S>(S(0)));
  static_assert(std::in_range<S>(std::numeric_limits<S>::max()));
  static_assert(std::in_range<S>(std::numeric_limits<S>::min()));
  // Signed target: value out of range via a wider unsigned source.
  static_assert(!std::in_range<S>(std::numeric_limits<U>::max()));
  // Unsigned target: negative signed value is out of range.
  static_assert(!std::in_range<U>(S(-1)));
  // Unsigned target: zero is in range.
  static_assert(std::in_range<U>(S(0)));
  static_assert(std::in_range<U>(std::numeric_limits<U>::max()));
  return true;
}

constexpr bool test() {
  // Branch 2 territory (sizeof <= sizeof(int)).
  test_same_sign<_BitInt(7), _BitInt(7)>();
  test_same_sign<unsigned _BitInt(7), unsigned _BitInt(7)>();
  test_same_sign<_BitInt(13), _BitInt(13)>();
  test_mixed_sign<_BitInt(7), unsigned _BitInt(7)>();
  test_mixed_sign<_BitInt(13), unsigned _BitInt(13)>();
  test_in_range<_BitInt(7), unsigned _BitInt(7)>();
  test_in_range<_BitInt(13), unsigned _BitInt(13)>();

  // Equal-sizeof-as-int boundary: signed _BitInt(32) can promote to int,
  // unsigned _BitInt(32) cannot (would lose the high bit), so it falls
  // into branch 3.
  test_same_sign<_BitInt(32), _BitInt(32)>();
  test_same_sign<unsigned _BitInt(32), unsigned _BitInt(32)>();
  test_mixed_sign<_BitInt(32), unsigned _BitInt(32)>();
  test_in_range<_BitInt(32), unsigned _BitInt(32)>();

  // Branch 3 territory (sizeof <= sizeof(long long)).
  test_same_sign<_BitInt(33), _BitInt(33)>();
  test_same_sign<_BitInt(63), _BitInt(63)>();
  test_same_sign<unsigned _BitInt(63), unsigned _BitInt(63)>();
  test_mixed_sign<_BitInt(33), unsigned _BitInt(33)>();
  test_mixed_sign<_BitInt(63), unsigned _BitInt(63)>();
  test_in_range<_BitInt(33), unsigned _BitInt(33)>();
  test_in_range<_BitInt(63), unsigned _BitInt(63)>();

  // Equal-sizeof-as-long-long boundary: _BitInt(64) signed promotes,
  // unsigned _BitInt(64) does not, so the mixed-sign case lands in
  // branch 4.
  test_same_sign<_BitInt(64), _BitInt(64)>();
  test_same_sign<unsigned _BitInt(64), unsigned _BitInt(64)>();
  test_mixed_sign<_BitInt(64), unsigned _BitInt(64)>();
  test_in_range<_BitInt(64), unsigned _BitInt(64)>();

#  if __BITINT_MAXWIDTH__ >= 128
  // Branch 4 territory (sizeof > sizeof(long long)).
  test_same_sign<_BitInt(65), _BitInt(65)>();
  test_same_sign<_BitInt(128), _BitInt(128)>();
  test_same_sign<unsigned _BitInt(128), unsigned _BitInt(128)>();
  test_mixed_sign<_BitInt(65), unsigned _BitInt(65)>();
  test_mixed_sign<_BitInt(128), unsigned _BitInt(128)>();
  test_in_range<_BitInt(65), unsigned _BitInt(65)>();
  test_in_range<_BitInt(128), unsigned _BitInt(128)>();
#  endif

#  if __BITINT_MAXWIDTH__ >= 200
  // Beyond __int128: verifies make_unsigned_t<_BitInt(N)> works on the
  // fallback path for widths with no builtin mapping.
  test_same_sign<_BitInt(200), _BitInt(200)>();
  test_same_sign<unsigned _BitInt(200), unsigned _BitInt(200)>();
  test_mixed_sign<_BitInt(200), unsigned _BitInt(200)>();
  test_in_range<_BitInt(200), unsigned _BitInt(200)>();
#  endif

  // Cross-width: narrow signed _BitInt vs wide unsigned builtin.
  // Negative source must be reported as less than any non-negative target.
  static_assert(std::cmp_less(_BitInt(7)(-1), 0ull));
  static_assert(std::cmp_less(_BitInt(13)(-1), 0u));
  static_assert(std::cmp_less(_BitInt(63)(-1), 0ull));
  // Cross-type round-trip equality.
  static_assert(std::cmp_equal(_BitInt(13)(42), 42));
  static_assert(std::cmp_equal(42, _BitInt(13)(42)));
  static_assert(std::cmp_equal(unsigned _BitInt(13)(42), 42u));

  return true;
}

#endif // TEST_HAS_EXTENSION(bit_int)

int main(int, char**) {
#if TEST_HAS_EXTENSION(bit_int)
  test();
  static_assert(test());
#endif
  return 0;
}
