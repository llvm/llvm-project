//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the statically rounded, integer-only implementation of
/// exp(x)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H

#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/frac128.h"
#include "src/__support/integer_literals.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/check/exp_exceptions.h"

namespace LIBC_NAMESPACE_DECL {

namespace shared {

namespace math {

namespace static_rounding {

using LIBC_NAMESPACE::operator""_u128;

// print(2+round(1/log(2), 128, RN));
// LSB(INV_LN2) = 2^-127
LIBC_INLINE_VAR constexpr Frac128 INV_LN2_F128 =
    Frac128(0xb8aa3b29'5c17f0bb'be87fed0'691d3e89_u128);

// exp(x) for x from 0 to (0b1111/2^4) = 15/16
// > for i from 0 to 15 do {
//   print(1+round(exp(1/(16-i)), 128, RN));
// };
// LSB(EXP_MID4[i]) = 2^-127
LIBC_INLINE_VAR constexpr Frac128 EXP_MID4[] = {
    Frac128(1u),
    Frac128(0x08415abb'e9a76bea'd8d00cf1'12e4d4a9_u128),
    Frac128(0x08d2ff22'4cc9cd75'daf50224'e6405613_u128),
    Frac128(0x097a3089'bcde8358'06617ab2'2577bd36_u128),
    Frac128(0x0a3c18b3'c8501827'a038960e'6554890c_u128),
    Frac128(0x0b1fac01'4ab4cc6e'f83ad1c5'0d2fb016_u128),
    Frac128(0x0c2e831f'eb8fa72a'da22cfa4'efa8e05c_u128),
    Frac128(0x0d763d9a'd0069cd7'c11c604c'2b558c0e_u128),
    Frac128(0x0f0add66'738b06a8'2a34e44f'7af67e71_u128),
    Frac128(0x110b022d'b7ae67ce'76b441c2'7035c6a1_u128),
    Frac128(0x13a8048b'71443b31'12f72650'b356e2df_u128),
    Frac128(0x1736d169'0604545d'44b774b0'16630f77_u128),
    Frac128(0x1c56ecf2'c5646740'b2bb19c5'bbfe54e1_u128),
    Frac128(0x245af1e1'f40c333b'3de1db4d'd55f29a7_u128),
    Frac128(0x32a36d8d'd1689885'3fb63793'fb4f42a0_u128),
    Frac128(0x53094c70'f034de4b'96ff7d5b'6f99fcd9_u128),
    Frac128(0xdbf0a8b1'45769535'5fb8ac40'4e7a79e4_u128),
};

// 128-bit polynomial approximation of e^x coefficients generated with Sollya:
// > P = fpminimax(exp(x), 12, [|1, 128...|], [0, 1/16], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(exp(x) - P(x), [0, 1/16]);
// 0x1.9295...p-110
// LSB(EXPF_COEFFS[i]) = 2^-128
LIBC_INLINE_VAR constexpr Frac128 EXP_COEFFS[] = {
    // degree-0 = 1, add back afterwards to reduce calc ops
    Frac128(0xffffffff'ffffffff'ffffffff'acc77512_u128),
    Frac128(0x80000000'00000000'0000015d'68ca804a_u128),
    Frac128(0x2aaaaaaa'aaaaaaaa'aaa8a818'b0e4f3c0_u128),
    Frac128(0x0aaaaaaa'aaaaaaaa'ac27ae0b'4b8e5377_u128),
    Frac128(0x02222222'22222221'7c98b384'6025fb89_u128),
    Frac128(0x005b05b0'5b05b088'd6cbde88'10bd52a0_u128),
    Frac128(0x000d00d0'0d00c797'c6f4c09c'21513949_u128),
    Frac128(0x0001a01a'01a12ab5'01d2062c'93e5cc17_u128),
    Frac128(0x00002e3b'c7332843'170d9ff8'520123e8_u128),
    Frac128(0x0000049f'954bbaf1'd7e43929'0c19789f_u128),
    Frac128(0x0000006b'8c001fdf'5a004af5'0d6f47b4_u128),
    Frac128(0x00000009'401bb484'e5aa7a08'b5a17885_u128),
};

// TODO: make 128-bit exp version work first, fast 64-bit and Ziv's test version
// later on.
// When rolling out the 64-bit fast path, take a look at expf also
//
// TODO: Ziv's test + pushing from Frac64 to full Frac128 pipeline on
// hard-to-round cases Ziv test by extracting the last 12 bits of the result
// (result & ((1u << 13) - 1)), cast into uint32_t, +4, and then | (1u << 12).
// If the result is > bound, we have a hard to round case, then dial to Frac128
// pipeline
//
// TODO: current implementation is mostly ported over from expf. Tricky input
// tests are failing. Debug.
//
// TODO: test against CORE-MATH

LIBC_INLINE double exp(double x, [[maybe_unused]] int rounding) {
  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  bool is_neg = xbits.is_neg();
  uint64_t x_val = xbits.uintval();
  uint64_t x_val_abs = xbits.abs().uintval();

  // TODO: dedupe? Ported over from the base LLVM-libc exp(x) implementation
  // with same checks, just without exceptions/errnos.
  //
  // x < log(2^-1075) or x >= 0x1.6232bdd7abcd3p+9 or |x| < 2^-53.
  if (LIBC_UNLIKELY(
          x_val >= 0xc0874910d52d3052 ||
          (x_val < 0xbca0000000000000 && x_val >= 0x40862e42fefa39f0) ||
          x_val < 0x3ca0000000000000)) {
    // |x| <= 2^-53
    if (x_val_abs <= 0x3ca0'0000'0000'0000ULL) {
      // exp(x) ~ 1 + x
      return 1 + x;
    }

    // x <= log(2^-1075) || x >= 0x1.6232bdd7abcd3p+9 or inf/nan.

    // x <= log(2^-1075) or -inf/nan
    if (x_val >= 0xc087'4910'd52d'3052ULL) {
      // exp(-Inf) = 0
      if (xbits.is_inf())
        return 0.0;

      // exp(nan) = nan
      if (xbits.is_nan())
        return x;

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (rounding == FE_UPWARD)
        return FPBits::min_subnormal().get_val();
#endif
      return 0.0;
    }

    // x >= round(log(MAX_NORMAL), D, RU) = 0x1.62e42fefa39fp+9 or +inf/nan
    // x is finite
    if (x_val < 0x7ff0'0000'0000'0000ULL) {
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO)
        return FPBits::max_normal().get_val();
#endif
    }
    // x is +inf or nan
    return x + FPBits::inf().get_val();
  }

  // Main calculations

  uint16_t x_e = xbits.get_biased_exponent();
  uint64_t x_s = xbits.get_mantissa();

  // The idea of the algorithm below is that:
  // For x = hi + mid + low, with:
  //  - hi is an integer
  //    - hi = N * ln(2)
  //  - mid * 2^4 is an integer
  //  - lo is the remainder bits
  // Then:
  //  exp(x) = 2^N * exp(mid) * exp(lo)
  // With this formula:
  //  - Multiplying by 2^N is exact and cheap, via adding into the exponent
  //  field
  //  - exp(mid) can be calculated via the LUT declared above
  //  - exp(lo) ~ 1 + lo + a0 * lo^2 + ...
  // Then we can construct exp(x) pretty easily, as hi, mid, lo bits can be
  // separate and be used independently, then we only need to reconstruct in the
  // final steps, which makes our life easier.
  //
  // Errors in computing in double-precision
  // TODO

  // Range reduction

  // Recall binary64:
  // 1 sign bit
  // 11 exp bits
  // 52 sig bits

  // add leading bit = 1
  x_s |= uint64_t(1) << FPBits::FRACTION_LEN;

  // shift to top 64 bits --> decimal point at hidden bit that we've added
  int x_e_unbiased = static_cast<int>(x_e) - FPBits::EXP_BIAS;

  // Shift left by 128 - 53 = 75 bits for the fractional part of the double
  // to align to the correct positions inside Frac128
  // LSB(x_s_frac) = 2^-75
  UInt128 x_s_shifted = static_cast<UInt128>(x_s) << 75;

  // Apply the exponent: shift so the binary point is at the hidden bit
  if (x_e_unbiased > 0) {
    x_s_shifted <<= x_e_unbiased;
  } else if (x_e_unbiased < 0) {
    x_s_shifted >>= -x_e_unbiased;
  }

  // LSB(x_s_frac) = 2^-75
  Frac128 x_s_frac(x_s_shifted);

  // LSB(x_ln2) = 2^-74 (product of 2^-75 * 2^127 / 2^128 ~ 2^-75 effectively,
  // but the integer part of x*log2(e) lands in the high word)
  Frac128 x_ln2 = x_s_frac * INV_LN2_F128;

  // We use the top 128-bit word of x_ln2 to extract N (the integer part of
  // x * log2(e)). The fractional part drives the polynomial/LUT evaluation.
  constexpr uint64_t FRAC_MASK = (uint64_t(1) << 52) - 1;
  uint64_t x_ln2_hi = x_ln2.val[1];

  uint64_t e_y, l2y_r_hi;
  uint32_t e_y_unbiased;

  // As Frac128 can't store the sign, we need to handle the sign separately:
  // - Both branches are computing floor(x * log2(e)).
  // - For negative x, we round up to the next multiple of 2^52, then clear the
  // last 52 bits.
  // - For positive x, we round down (just clear) the last 52 bits.
  //
  // Then, l2y_r_hi is the remainder of x * log2(e) after removing the integer
  // part, which is used to look up EXP_MID4 and compute exp(lo) - 1.
  //
  // e_y_unbiased is biased exponent field, but already bit-positioned to the
  // exponent field of the double representation.
  if (LIBC_UNLIKELY(is_neg)) {
    e_y = (x_ln2_hi + FRAC_MASK) & ~FRAC_MASK;
    l2y_r_hi = e_y - x_ln2_hi;
    e_y_unbiased = (FPBits::EXP_BIAS << 20) - static_cast<uint32_t>(e_y >> 43);
  } else {
    e_y = x_ln2_hi & ~FRAC_MASK;
    l2y_r_hi = x_ln2_hi - e_y;
    e_y_unbiased = (FPBits::EXP_BIAS << 20) + static_cast<uint32_t>(e_y >> 43);
  }

  uint32_t k = static_cast<uint32_t>(e_y >> 52);
  int d = static_cast<int>(k) - FPBits::EXP_BIAS;

  // d >= 53 --> k >= 1076
  // --> guaranteed to be below 2^-1074
  //
  // underflow
  if (LIBC_UNLIKELY(is_neg && d >= 53)) {
    return 0.0;
  }

  // Extract the 4 mid bits (bits [51:48] of l2y_r_hi) for LUT index
  uint16_t x_mid = static_cast<uint16_t>((l2y_r_hi >> 48) & 0xf);

  // Extract the low 48 bits for the polynomial approximation of exp(lo).
  // LSB(x_lo) = 2^-48
  uint64_t x_lo = l2y_r_hi & ((uint64_t(1) << 48) - 1);

  // Shift left to move all bits to the high part
  // LSB(x_lo_frac) = 2^-128
  Frac128 x_lo_frac(static_cast<UInt128>(x_lo) << 80);

  Frac128 p =
      x_lo_frac * fputil::polyeval(x_lo_frac, EXP_COEFFS[0], EXP_COEFFS[1],
                                   EXP_COEFFS[2], EXP_COEFFS[3], EXP_COEFFS[4],
                                   EXP_COEFFS[5], EXP_COEFFS[6], EXP_COEFFS[7],
                                   EXP_COEFFS[8], EXP_COEFFS[9], EXP_COEFFS[10],
                                   EXP_COEFFS[11]);

  // With:
  //  - p = exp(lo) - 1 --> exp(lo) = p + 1
  //  - mid_val = exp(mid) = EXP_MID4[x_mid]
  // We have:
  //  exp(mid) * exp(lo) = mid_val * (p + 1)
  // (Workaround because we're dealing with fractional representation of things)
  Frac128 mid_val = EXP_MID4[x_mid];
  Frac128 result128 = mid_val * p + mid_val;

  // We're computing with errors < worst-case errors, so tie-rounding never
  // happens. Hence, round-to-nearest, tie-to-even is equivalent to
  // round-to-nearest, tie-to-away. Which is what we're implementing below
  // in the following order:
  //
  // 1. Shift so that the rounding bit is at bit-0
  // 2. Add 1 for rounding
  // 3. Perform another shift by 1
  // 4. Depending on the rounding modes, adjust accordingly:
  //  a. Add 1 if rounding-up (0 if not)
  //  b. Add e_y_unbiased to the result (0 if the result is subnormal)

  uint32_t shift_length = 72;
  uint32_t leading_one = 0;

  // subnormal
  if (LIBC_UNLIKELY(is_neg && d >= 0)) {
    e_y_unbiased = 0;
    leading_one = 1 << (52 - d);

    // In the below shifts, we're shifting by (shift_length + 1) at max, while
    // shift_length is already 72, and if d = 52 --> shift_length + d = 124,
    // and we'll shift by whole 128 bits, which is undefined behavior in C++.
    //
    // So, we'll truncate the last 2 bits.
    if (d >= 51) {
      d -= 2;
      result128.val[1] >>= 2;
    }

    shift_length += d + 1;
  }

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  uint64_t result = (static_cast<uint64_t>(result128.val[1] >> shift_length) +
                     (static_cast<uint64_t>(leading_one) + 1));
  result >>= 1;
  result += static_cast<uint64_t>(e_y_unbiased) << 32;

  return cpp::bit_cast<double>(result);
#else  // !LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  if (rounding == FE_TONEAREST) {
    uint64_t result = (static_cast<uint64_t>(result128.val[1] >> shift_length) +
                       (static_cast<uint64_t>(leading_one) + 1));
    result >>= 1;
    result += static_cast<uint64_t>(e_y_unbiased) << 32;

    return cpp::bit_cast<double>(result);
  }

  uint64_t should_round_up = 0;

  if (LIBC_UNLIKELY(rounding == FE_UPWARD)) {
    uint64_t round_up_mask = (uint64_t(1) << (shift_length + 1)) - 1;
    should_round_up =
        static_cast<uint64_t>((result128.val[1] & round_up_mask) != 0);
  }

  uint64_t result =
      (static_cast<uint64_t>(result128.val[1] >> (shift_length + 1)) +
       should_round_up + (static_cast<uint64_t>(leading_one) >> 1));
  result += static_cast<uint64_t>(e_y_unbiased) << 32;

  return cpp::bit_cast<double>(result);
#endif // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
}

} // namespace static_rounding

} // namespace math

} // namespace shared

namespace math {
namespace integer_eval {

LIBC_INLINE double exp(double x) {
  return shared::math::static_rounding::exp(x, FE_TONEAREST);
}

} // namespace integer_eval
} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
