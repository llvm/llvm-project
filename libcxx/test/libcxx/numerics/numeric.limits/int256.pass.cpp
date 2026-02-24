//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test std::numeric_limits specialization for __int256_t / __uint256_t.
//
// The generic __libcpp_numeric_limits<_Tp, true> template handles all
// arithmetic types, including __int256_t and __uint256_t. This test verifies
// that the specialization produces correct values for all properties.

#include <cstddef>
#include <limits>
#include <type_traits>

#include "test_macros.h"

#ifdef TEST_HAS_NO_INT256
int main(int, char**) { return 0; }
#else

// ========================================================================
// Static properties (compile-time)
// ========================================================================

// --- is_specialized ---
static_assert(std::numeric_limits<__int256_t>::is_specialized, "");
static_assert(std::numeric_limits<__uint256_t>::is_specialized, "");
static_assert(std::numeric_limits<const __int256_t>::is_specialized, "");
static_assert(std::numeric_limits<volatile __uint256_t>::is_specialized, "");
static_assert(std::numeric_limits<const volatile __int256_t>::is_specialized, "");

// --- is_signed ---
static_assert(std::numeric_limits<__int256_t>::is_signed, "");
static_assert(!std::numeric_limits<__uint256_t>::is_signed, "");

// --- is_integer, is_exact ---
static_assert(std::numeric_limits<__int256_t>::is_integer, "");
static_assert(std::numeric_limits<__uint256_t>::is_integer, "");
static_assert(std::numeric_limits<__int256_t>::is_exact, "");
static_assert(std::numeric_limits<__uint256_t>::is_exact, "");

// --- radix ---
static_assert(std::numeric_limits<__int256_t>::radix == 2, "");
static_assert(std::numeric_limits<__uint256_t>::radix == 2, "");

// --- digits ---
// __int256_t: 256 bits - 1 sign bit = 255 value bits
// __uint256_t: 256 bits, all value bits
static_assert(std::numeric_limits<__int256_t>::digits == 255, "");
static_assert(std::numeric_limits<__uint256_t>::digits == 256, "");

// --- digits10 ---
// digits10 = floor(digits * log10(2))
// For __int256_t:  floor(255 * 0.30103) = floor(76.76) = 76
// For __uint256_t: floor(256 * 0.30103) = floor(77.06) = 77
static_assert(std::numeric_limits<__int256_t>::digits10 == 76, "");
static_assert(std::numeric_limits<__uint256_t>::digits10 == 77, "");

// --- max_digits10 ---
static_assert(std::numeric_limits<__int256_t>::max_digits10 == 0, "");
static_assert(std::numeric_limits<__uint256_t>::max_digits10 == 0, "");

// --- is_bounded ---
static_assert(std::numeric_limits<__int256_t>::is_bounded, "");
static_assert(std::numeric_limits<__uint256_t>::is_bounded, "");

// --- is_modulo ---
// Signed: not modulo (overflow is UB). Unsigned: modulo (wraps).
static_assert(!std::numeric_limits<__int256_t>::is_modulo, "");
static_assert(std::numeric_limits<__uint256_t>::is_modulo, "");

// --- has_infinity, has_quiet_NaN, etc. ---
static_assert(!std::numeric_limits<__int256_t>::has_infinity, "");
static_assert(!std::numeric_limits<__uint256_t>::has_infinity, "");
static_assert(!std::numeric_limits<__int256_t>::has_quiet_NaN, "");
static_assert(!std::numeric_limits<__uint256_t>::has_quiet_NaN, "");
static_assert(!std::numeric_limits<__int256_t>::has_signaling_NaN, "");
static_assert(!std::numeric_limits<__uint256_t>::has_signaling_NaN, "");

// --- is_iec559 ---
static_assert(!std::numeric_limits<__int256_t>::is_iec559, "");
static_assert(!std::numeric_limits<__uint256_t>::is_iec559, "");

// --- exponent fields ---
static_assert(std::numeric_limits<__int256_t>::min_exponent == 0, "");
static_assert(std::numeric_limits<__int256_t>::max_exponent == 0, "");
static_assert(std::numeric_limits<__int256_t>::min_exponent10 == 0, "");
static_assert(std::numeric_limits<__int256_t>::max_exponent10 == 0, "");

// --- round_style ---
static_assert(std::numeric_limits<__int256_t>::round_style == std::round_toward_zero, "");
static_assert(std::numeric_limits<__uint256_t>::round_style == std::round_toward_zero, "");

// --- Relationship to __int128 ---
static_assert(std::numeric_limits<__int256_t>::digits == 2 * std::numeric_limits<__int128_t>::digits + 1, "");
static_assert(std::numeric_limits<__uint256_t>::digits == 2 * std::numeric_limits<__uint128_t>::digits, "");

// ========================================================================
// Runtime value checks
// ========================================================================

int main(int, char**) {
  // --- unsigned min/max ---
  {
    __uint256_t umin = std::numeric_limits<__uint256_t>::min();
    __uint256_t umax = std::numeric_limits<__uint256_t>::max();
    __uint256_t ulow = std::numeric_limits<__uint256_t>::lowest();

    // min() for unsigned is 0
    if (umin != 0)
      return 1;

    // max() is all-ones (2^256 - 1)
    if (umax != ~(__uint256_t)0)
      return 2;

    // lowest() == min() for integers
    if (ulow != umin)
      return 3;

    // max + 1 wraps to 0 (unsigned modulo)
    __uint256_t wrapped = umax + 1;
    if (wrapped != 0)
      return 4;
  }

  // --- signed min/max ---
  {
    __int256_t smin = std::numeric_limits<__int256_t>::min();
    __int256_t smax = std::numeric_limits<__int256_t>::max();
    __int256_t slow = std::numeric_limits<__int256_t>::lowest();

    // min() is negative (sign bit set)
    if (smin >= 0)
      return 5;

    // max() is positive
    if (smax <= 0)
      return 6;

    // lowest() == min() for integers
    if (slow != smin)
      return 7;

    // min() == -(2^255)
    // Verify by checking that min() has only the MSB set when viewed as unsigned
    __uint256_t umin_bits    = (__uint256_t)smin;
    __uint256_t expected_msb = (__uint256_t)1 << 255;
    if (umin_bits != expected_msb)
      return 8;

    // max() == 2^255 - 1
    // All bits except MSB are set
    __uint256_t umax_bits = (__uint256_t)smax;
    if (umax_bits != (expected_msb - 1))
      return 9;

    // min + max == -1 (two's complement identity)
    if (smin + smax != -1)
      return 10;
  }

  // --- epsilon, denorm_min, infinity, NaN are all zero for integers ---
  {
    if (std::numeric_limits<__int256_t>::epsilon() != 0)
      return 11;
    if (std::numeric_limits<__int256_t>::round_error() != 0)
      return 12;
    if (std::numeric_limits<__int256_t>::infinity() != 0)
      return 13;
    if (std::numeric_limits<__int256_t>::quiet_NaN() != 0)
      return 14;
    if (std::numeric_limits<__int256_t>::signaling_NaN() != 0)
      return 15;
    if (std::numeric_limits<__int256_t>::denorm_min() != 0)
      return 16;
  }

  // --- const/volatile qualifiers preserve behavior ---
  {
    if (std::numeric_limits<const __uint256_t>::max() != std::numeric_limits<__uint256_t>::max())
      return 17;
    if (std::numeric_limits<volatile __int256_t>::min() != std::numeric_limits<__int256_t>::min())
      return 18;
    if (std::numeric_limits<const volatile __uint256_t>::digits != 256)
      return 19;
  }

  // --- Cross-check with __int128 ---
  {
    // max(__uint256_t) > max(__uint128_t)
    __uint256_t u256_max = std::numeric_limits<__uint256_t>::max();
    __uint128_t u128_max = std::numeric_limits<__uint128_t>::max();
    if (u256_max <= (__uint256_t)u128_max)
      return 20;

    // The upper 128 bits of max(__uint256_t) should be max(__uint128_t)
    __uint128_t upper = (__uint128_t)(u256_max >> 128);
    if (upper != u128_max)
      return 21;
  }

  return 0;
}
#endif
