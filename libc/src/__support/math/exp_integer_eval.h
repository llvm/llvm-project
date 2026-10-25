//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integer-only implementation of exp.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H

#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/frac128.h"
#include "src/__support/frac64.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp_integer_utils.h"

// TODO: share *_integer_utils.h with exp2 the same way we did with expf/exp2f.

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace static_rounding {

LIBC_ALWAYS_INLINE uint64_t exp_bits(uint64_t x_u,
                                     [[maybe_unused]] int rounding) {
  using FPBits = fputil::FPBits<double>;

  uint64_t x_abs = x_u & 0x7fff'ffff'ffff'ffff;

  if (x_abs > FPBits::inf().uintval())
    return x_u | (uint64_t(1) << (FPBits::FRACTION_LEN - 1));

  if (LIBC_UNLIKELY(
          x_u >= 0xc087'4910'd52d'3052 ||
          (x_u < 0xbca0'0000'0000'0000 && x_u >= 0x4086'2e42'fefa'39f0) ||
          x_u <= 0x3ca0'0000'0000'0000)) {
    // |x| <= 2^-53
    if (x_abs <= 0x3ca0'0000'0000'0000) {
      uint64_t one = FPBits::one().uintval();
      if (x_abs == 0)
        return one;
      bool is_neg = (x_u >> 63) != 0;
      if (x_abs == 0x3ca0'0000'0000'0000) {
        if (rounding == FE_TONEAREST || (rounding == FE_UPWARD && !is_neg) ||
            ((rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO) && is_neg))
          return is_neg ? one - 1 : one + 1;
      }
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (rounding == FE_UPWARD && !is_neg)
        return one + 1;
      if ((rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO) && is_neg)
        return one - 1;
#endif
      return one;
    }

    // x <= log(2^-1075) || x >= 0x1.6232bdd7abcd3p+9 or inf/nan.

    // x <= log(2^-1075) or -inf/nan
    if (x_u >= 0xc087'4910'd52d'3052) {
      // exp(-Inf) = 0
      if (x_abs == FPBits::inf().uintval())
        return FPBits::zero().uintval();

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (rounding == FE_UPWARD)
        return FPBits::min_subnormal().uintval();
#endif
      return FPBits::zero().uintval();
    }

    // x >= round(log(MAX_NORMAL), D, RU) = 0x1.62e42fefa39fp+9 or +inf/nan
    // x is finite
    if (x_u < 0x7ff0'0000'0000'0000) {
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO)
        return FPBits::max_normal().uintval();
#endif
    }

    // x is +inf or nan
    return FPBits::inf().uintval();
  }

  // Main calculations

  bool is_neg = (x_u >> 63) != 0;
  uint32_t x_e = static_cast<uint32_t>(x_abs >> FPBits::FRACTION_LEN);
  uint64_t x_s = x_abs & FPBits::FRACTION_MASK;

  // The idea of the algorithm below is that:
  // Let:
  //  - y = x / ln(2) --> exp(x) = 2^y.
  //  - y = hi + mid + low
  //
  // With:
  //  - hi is an integer
  //  - mid * 2^4 is an integer
  //  - lo are the remainder bits
  //
  // Then:
  //  exp(x) = 2^hi * 2^mid * 2^lo
  //
  // With this formula:
  //  - Multiplying by 2^hi is exact and cheap, via adding into the exponent
  //  field
  //  - 2^mid can be calculated via the LUT declared above
  //  - 2^lo ~ 1 + lo + a0 * lo^2 + ...
  //
  // Then we can construct exp(x) pretty easily, as hi, mid, lo bits can be
  // separate and be used independently, then we only need to reconstruct in the
  // final steps, which makes our life easier.

  // Range reduction

  // binary64 recall:
  // 1 sign bit
  // 11 exp bits
  // 52 sig bits

  uint64_t x_s_shifted = (x_s << 11) | (1ull << 63);
  // LSB(x_s_frac) = 2^-52
  Frac64 x_s_frac(x_s_shifted);

  Frac64 x_ln2 = x_s_frac * LOG2_E_F64;
  uint64_t x_ln2_bits = x_ln2.val[0];

  uint64_t k = 0;
  uint64_t frac_bits = 0;
  int shift = 0;

  int x_e_unbiased = static_cast<int>(x_e) - FPBits::EXP_BIAS;
  if (x_e_unbiased >= -1) {
    shift = 62 - x_e_unbiased;
    k = x_ln2_bits >> shift;
    frac_bits = x_ln2_bits << (64 - shift);
  } else {
    k = 0;
    shift = -x_e_unbiased - 2;
    frac_bits = (shift < 64) ? (x_ln2_bits >> shift) : 0;
  }

  // As Frac64/Frac128 can't store the sign, we need to handle the sign
  // separately:
  // - Both branches are computing floor(x * log2(e)).
  // - For negative x, we round up to the next multiple of 2^52, then clear
  // the last 52 bits.
  // - For positive x, we round down (just clear) the last 52 bits.
  //
  // Then, l2y_r_hi is the remainder of x * log2(e) after removing the
  // integer part, which is used to look up EXP_MID and compute 2^lo.
  //
  // e_y is the biased exponent field, positioned for the final bit assembly.
  uint64_t l2y_r_hi;
  uint64_t e_y;

  if (LIBC_UNLIKELY(is_neg)) {
    if (frac_bits != 0) {
      k = k + 1;
      l2y_r_hi = ~frac_bits + 1; // 1 - r
    } else {
      l2y_r_hi = 0;
    }
    e_y = (FPBits::EXP_BIAS << 20) - static_cast<uint32_t>(k << 20);
  } else {
    l2y_r_hi = frac_bits;
    e_y = (FPBits::EXP_BIAS << 20) + static_cast<uint32_t>(k << 20);
  }

  int d = static_cast<int>(k) - FPBits::EXP_BIAS;

  // d >= 53 --> k >= 1076
  // --> guaranteed to be below 2^-1074
  //
  // underflow
  if (LIBC_UNLIKELY(is_neg && d >= 53)) {
    return 0.0;
  }

  // Extract the top 4 fractional bits for the LUT index.
  uint16_t x_mid = static_cast<uint16_t>((l2y_r_hi >> 60) & 0xf);

  // The remaining 60 bits are the polynomial input.
  uint64_t x_lo = l2y_r_hi & ((1ull << 60) - 1);

  // Fast path: 64-bit calculations first

  // Don't << 4, as the polynomial approximation is correct in range [0, 1/16],
  // and x_lo is already in that range.
  // LSB(x_lo_frac) = 2^-64
  Frac64 x_lo_frac(x_lo);

  Frac64 p = x_lo_frac * fputil::polyeval(x_lo_frac, EXP_64_COEFFS[0],
                                          EXP_64_COEFFS[1], EXP_64_COEFFS[2],
                                          EXP_64_COEFFS[3], EXP_64_COEFFS[4],
                                          EXP_64_COEFFS[5]);

  // With:
  //  - p = 2^lo - 1 --> 2^lo = p + 1
  //  - mid_val = 2^mid = EXP_MID[x_mid]
  // We have:
  //  2^mid * 2^lo = mid_val * (p + 1)
  // The same applies for both of the 64-bit and 128-bit paths
  // (Workaround because we're dealing with fractional representation of things)
  Frac64 mid_val = EXP_MID[x_mid].to_frac64();
  Frac64 result = fputil::multiply_add(mid_val, p, mid_val);

  uint64_t result_bits = result.val[0] << 1;

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  return exp_handle_rounding(result, is_neg, d, e_y, rounding);
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

  // Rounding test
  constexpr uint32_t LAST_BITS = 12;
  constexpr uint32_t ROUNDING_ERROR = (1ull << 11);
  uint32_t result_last_bits =
      static_cast<uint32_t>(result_bits & ((1u << LAST_BITS) - 1));
  bool is_hard;
  if (rounding == FE_TONEAREST) {
    uint32_t rounded_lo =
        (result_last_bits + (1u << (LAST_BITS - 1)) - ROUNDING_ERROR) >>
        LAST_BITS;
    uint32_t rounded_hi =
        (result_last_bits + (1u << (LAST_BITS - 1)) + ROUNDING_ERROR) >>
        LAST_BITS;
    is_hard = rounded_lo != rounded_hi;
  } else {
    is_hard = result_last_bits <= ROUNDING_ERROR ||
              result_last_bits >= (1u << LAST_BITS) - ROUNDING_ERROR;
  }

  if (LIBC_LIKELY(!is_hard)) {
    return exp_handle_rounding(result, is_neg, d, e_y, rounding);
  }

  // Dial back to 128-bit path for hard-to-round cases
  return exp_accurate_path(x_s_shifted, x_e_unbiased, is_neg, rounding);
}

LIBC_ALWAYS_INLINE double exp(double x, [[maybe_unused]] int rounding) {
  return cpp::bit_cast<double>(exp_bits(cpp::bit_cast<uint64_t>(x), rounding));
}

} // namespace static_rounding

namespace integer_eval {

LIBC_ALWAYS_INLINE double exp(double x) {
  return static_rounding::exp(x, FE_TONEAREST);
}

} // namespace integer_eval

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
