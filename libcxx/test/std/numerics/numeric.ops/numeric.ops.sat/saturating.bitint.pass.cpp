//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <numeric>

// add_sat, sub_sat, mul_sat, div_sat, saturate_cast applied to _BitInt(N).
//
// After [libc++] recognized _BitInt as an integer type in
// __type_traits/integer_traits.h, these functions silently started
// accepting _BitInt arguments. Saturation at min/max depends on
// numeric_limits<_BitInt(N)>::min/max being correct, which requires the
// digits10 fix from #193002 for odd widths.
//
// Widths covered:
//   -  _BitInt(13):  odd narrow width, signed range -4096..4095.
//                    Exercises fixed digits10 for saturation clamp.
//   -  _BitInt(64):  equal to long long, integer_traits boundary.
//   -  _BitInt(128): matches __int128 on targets that support it.
//   -  _BitInt(200): beyond __int128 (optional via __BITINT_MAXWIDTH__).

#include <cassert>
#include <limits>
#include <numeric>

#include "test_macros.h"

#if TEST_HAS_EXTENSION(bit_int)

template <class T>
constexpr bool test_signed_add_sub() {
  constexpr T min_v = std::numeric_limits<T>::min();
  constexpr T max_v = std::numeric_limits<T>::max();

  // Basic: no overflow.
  assert(std::add_sat(T(1), T(2)) == T(3));
  assert(std::add_sat(T(-1), T(1)) == T(0));
  assert(std::sub_sat(T(5), T(3)) == T(2));
  assert(std::sub_sat(T(-1), T(-1)) == T(0));

  // Positive overflow clamps to max.
  assert(std::add_sat(max_v, T(1)) == max_v);
  assert(std::add_sat(T(1), max_v) == max_v);
  assert(std::add_sat(max_v, max_v) == max_v);

  // Negative overflow clamps to min.
  assert(std::add_sat(min_v, T(-1)) == min_v);
  assert(std::add_sat(T(-1), min_v) == min_v);
  assert(std::add_sat(min_v, min_v) == min_v);

  // sub_sat positive overflow (x >= 0, y < 0).
  assert(std::sub_sat(max_v, T(-1)) == max_v);
  assert(std::sub_sat(max_v, min_v) == max_v);

  // sub_sat negative overflow (x < 0, y > 0).
  assert(std::sub_sat(min_v, T(1)) == min_v);
  assert(std::sub_sat(min_v, max_v) == min_v);

  return true;
}

template <class T>
constexpr bool test_unsigned_add_sub() {
  constexpr T max_v = std::numeric_limits<T>::max();

  // Basic.
  assert(std::add_sat(T(1), T(2)) == T(3));
  assert(std::sub_sat(T(5), T(3)) == T(2));

  // Upper clamp.
  assert(std::add_sat(max_v, T(1)) == max_v);
  assert(std::add_sat(T(1), max_v) == max_v);
  assert(std::add_sat(max_v, max_v) == max_v);

  // Lower clamp (wrap-to-zero on unsigned).
  assert(std::sub_sat(T(0), T(1)) == T(0));
  assert(std::sub_sat(T(0), max_v) == T(0));
  assert(std::sub_sat(T(3), T(5)) == T(0));

  return true;
}

template <class T>
constexpr bool test_signed_mul_div() {
  constexpr T min_v = std::numeric_limits<T>::min();
  constexpr T max_v = std::numeric_limits<T>::max();

  // Basic mul.
  assert(std::mul_sat(T(2), T(3)) == T(6));
  assert(std::mul_sat(T(-2), T(3)) == T(-6));

  // Overflow to max.
  assert(std::mul_sat(max_v, T(2)) == max_v);
  assert(std::mul_sat(T(-1), min_v) == max_v); // -(-min) overflows to +max
  assert(std::mul_sat(min_v, T(-1)) == max_v);

  // Overflow to min.
  assert(std::mul_sat(max_v, T(-2)) == min_v);
  assert(std::mul_sat(T(-2), max_v) == min_v);

  // div_sat: regular values.
  assert(std::div_sat(T(6), T(3)) == T(2));
  assert(std::div_sat(T(7), T(3)) == T(2));
  assert(std::div_sat(T(-6), T(3)) == T(-2));

  // The one signed division overflow case: INT_MIN / -1.
  assert(std::div_sat(min_v, T(-1)) == max_v);

  return true;
}

template <class T>
constexpr bool test_unsigned_mul_div() {
  constexpr T max_v = std::numeric_limits<T>::max();

  assert(std::mul_sat(T(2), T(3)) == T(6));
  assert(std::mul_sat(max_v, T(2)) == max_v); // clamp
  assert(std::mul_sat(T(0), max_v) == T(0));
  assert(std::mul_sat(max_v, max_v) == max_v);

  assert(std::div_sat(T(10), T(3)) == T(3));
  assert(std::div_sat(max_v, T(1)) == max_v);
  return true;
}

template <class S, class U>
constexpr bool test_saturate_cast() {
  constexpr S s_min = std::numeric_limits<S>::min();
  constexpr S s_max = std::numeric_limits<S>::max();
  constexpr U u_max = std::numeric_limits<U>::max();

  // Same-type: no clamp.
  assert(std::saturate_cast<S>(S(0)) == S(0));
  assert(std::saturate_cast<S>(s_max) == s_max);
  assert(std::saturate_cast<S>(s_min) == s_min);
  assert(std::saturate_cast<U>(U(0)) == U(0));
  assert(std::saturate_cast<U>(u_max) == u_max);

  // Signed -> unsigned: negative clamps to zero.
  assert(std::saturate_cast<U>(S(-1)) == U(0));
  assert(std::saturate_cast<U>(s_min) == U(0));
  assert(std::saturate_cast<U>(S(1)) == U(1));

  // Unsigned -> signed: overflow clamps to s_max.
  assert(std::saturate_cast<S>(u_max) == s_max);

  return true;
}

constexpr bool test() {
  // Guaranteed width (<= 64).
  test_signed_add_sub<_BitInt(13)>();
  test_unsigned_add_sub<unsigned _BitInt(13)>();
  test_signed_mul_div<_BitInt(13)>();
  test_unsigned_mul_div<unsigned _BitInt(13)>();
  test_saturate_cast<_BitInt(13), unsigned _BitInt(13)>();

  test_signed_add_sub<_BitInt(64)>();
  test_unsigned_add_sub<unsigned _BitInt(64)>();
  test_signed_mul_div<_BitInt(64)>();
  test_unsigned_mul_div<unsigned _BitInt(64)>();
  test_saturate_cast<_BitInt(64), unsigned _BitInt(64)>();

  // Cross-width saturate_cast: wide source clamped into narrow target.
  {
    using S13 = _BitInt(13);
    using S64 = _BitInt(64);
    using U13 = unsigned _BitInt(13);
    using U64 = unsigned _BitInt(64);

    // wide signed -> narrow signed
    assert(std::saturate_cast<S13>(std::numeric_limits<S64>::max()) == std::numeric_limits<S13>::max());
    assert(std::saturate_cast<S13>(std::numeric_limits<S64>::min()) == std::numeric_limits<S13>::min());
    // wide unsigned -> narrow signed
    assert(std::saturate_cast<S13>(std::numeric_limits<U64>::max()) == std::numeric_limits<S13>::max());
    // wide signed -> narrow unsigned
    assert(std::saturate_cast<U13>(std::numeric_limits<S64>::min()) == U13{0});
    assert(std::saturate_cast<U13>(std::numeric_limits<S64>::max()) == std::numeric_limits<U13>::max());
    // exact-fit no clamp
    assert(std::saturate_cast<S64>(S13{-1}) == S64{-1});
    assert(std::saturate_cast<U64>(U13{42}) == U64{42});
  }

#  if __BITINT_MAXWIDTH__ >= 128
  test_signed_add_sub<_BitInt(128)>();
  test_unsigned_add_sub<unsigned _BitInt(128)>();
  test_signed_mul_div<_BitInt(128)>();
  test_unsigned_mul_div<unsigned _BitInt(128)>();
  test_saturate_cast<_BitInt(128), unsigned _BitInt(128)>();
#  endif

#  if __BITINT_MAXWIDTH__ >= 200
  // Beyond __int128: exercises the overflow-detection fallback on widths
  // with no builtin add/sub/mul_sat mapping.
  test_signed_add_sub<_BitInt(200)>();
  test_unsigned_add_sub<unsigned _BitInt(200)>();
  test_signed_mul_div<_BitInt(200)>();
  test_unsigned_mul_div<unsigned _BitInt(200)>();
  test_saturate_cast<_BitInt(200), unsigned _BitInt(200)>();

  // Cross-width between 128- and 200-bit widths.
  {
    using S200 = _BitInt(200);
    using S128 = _BitInt(128);
    assert(std::saturate_cast<S128>(std::numeric_limits<S200>::max()) == std::numeric_limits<S128>::max());
    assert(std::saturate_cast<S128>(std::numeric_limits<S200>::min()) == std::numeric_limits<S128>::min());
  }
#  endif

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
