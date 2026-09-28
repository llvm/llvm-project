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
#include "src/__support/CPP/type_traits/enable_if.h"
#include "src/__support/CPP/type_traits/is_same.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/frac128.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp_integer_constants.h" // LUTs

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace static_rounding {

// Round the fractional result and combine it with its exponent.
template <typename TFrac, typename TUInt>
LIBC_INLINE typename cpp::enable_if<cpp::is_same<TFrac, Frac64>::value ||
                                        cpp::is_same<TFrac, Frac128>::value,
                                    double>::type
exp_handle_rounding(TFrac result_frac, bool is_neg, int d, TUInt e_y,
                    [[maybe_unused]] int rounding) {
  constexpr bool IS_FAST_PATH = cpp::is_same<TFrac, Frac64>::value;

  uint32_t shift_length = 11;
  uint64_t leading_one = 0;

  // subnormal
  if (LIBC_UNLIKELY(is_neg && d >= 0)) {
    e_y = 0;
    leading_one = uint64_t(1) << (52 - d);

    // Truncate the last 2 bits to avoid undefined behavior when shifting by 64
    // bits
    if (d >= 51) {
      d -= 2;
      if constexpr (IS_FAST_PATH)
        result_frac.val[0] >>= 2;
      else
        result_frac.val[1] >>= 2;
    }

    shift_length += d + 1;
  }

  // Get the bits, discarding the leading bit
  auto frac_bits = [&]() -> uint64_t {
    if constexpr (IS_FAST_PATH)
      return result_frac.val[0] << 1;
    else
      return (result_frac.val[1] << 1) | (result_frac.val[0] >> 63);
  };

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  TUInt result =
      (static_cast<TUInt>(frac_bits() >> shift_length) + (leading_one + 1));
  result >>= 1;
  result += static_cast<TUInt>(e_y) << 32;

  return cpp::bit_cast<double>(result);
#else  // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  if (rounding == FE_TONEAREST) {
    TUInt result =
        (static_cast<TUInt>(frac_bits() >> shift_length) + (leading_one + 1));
    result >>= 1;
    result += static_cast<TUInt>(e_y) << 32;

    return cpp::bit_cast<double>(result);
  }

  TUInt should_round_up = 0;

  if (LIBC_UNLIKELY(rounding == FE_UPWARD)) {
    uint64_t round_up_mask = (uint64_t(1) << (shift_length + 1)) - 1;
    bool has_remainder = (frac_bits() & round_up_mask) != 0;
    if constexpr (!IS_FAST_PATH) {
      has_remainder = has_remainder || (result_frac.val[0] != 0);
    }
    should_round_up = static_cast<TUInt>(has_remainder);
  }

  TUInt result = (static_cast<TUInt>(frac_bits() >> (shift_length + 1)) +
                  should_round_up + (leading_one >> 1));
  result += static_cast<TUInt>(e_y) << 32;

  return cpp::bit_cast<double>(result);
#endif // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
}

LIBC_INLINE double exp_accurate_path(uint64_t x_s_shifted, int x_e_unbiased,
                                     bool is_neg, int rounding) {
  using FPBits = typename fputil::FPBits<double>;

  // Recalculate everything in 128-bit precision, with the same idea as the
  // 64-bit path.

  Frac128 x_s_frac({0, x_s_shifted});
  Frac128 x_ln2 = x_s_frac * INV_LN2_F128;

  uint64_t k = 0;
  Frac128 l2y_r;
  if (x_e_unbiased >= -1) {
    int shift = 62 - x_e_unbiased;
    k = (x_ln2 >> (64 + shift)).val[0];
    l2y_r = x_ln2 << (64 - shift);
  } else {
    int shift = -x_e_unbiased - 2;
    l2y_r = (shift < 128) ? (x_ln2 >> shift) : Frac128{};
  }

  if (LIBC_UNLIKELY(is_neg)) {
    if (l2y_r.val[0] != 0 || l2y_r.val[1] != 0) {
      ++k;
      l2y_r = ~l2y_r + Frac128(1);
    }
  }

  uint64_t e_y;
  if (is_neg)
    e_y = (FPBits::EXP_BIAS << 20) - static_cast<uint32_t>(k << 20);
  else
    e_y = (FPBits::EXP_BIAS << 20) + static_cast<uint32_t>(k << 20);

  int d = static_cast<int>(k) - FPBits::EXP_BIAS;

  if (LIBC_UNLIKELY(is_neg && d >= 53))
    return 0.0;

  uint16_t x_mid = static_cast<uint16_t>((l2y_r.val[1] >> 60) & 0xf);

  Frac128 x_lo_frac = l2y_r;
  x_lo_frac.val[1] &= 0x0fff'ffff'ffff'ffffULL;

  Frac128 p =
      x_lo_frac * fputil::polyeval(x_lo_frac, EXP_COEFFS[0], EXP_COEFFS[1],
                                   EXP_COEFFS[2], EXP_COEFFS[3], EXP_COEFFS[4],
                                   EXP_COEFFS[5], EXP_COEFFS[6], EXP_COEFFS[7],
                                   EXP_COEFFS[8], EXP_COEFFS[9], EXP_COEFFS[10],
                                   EXP_COEFFS[11]);

  Frac128 mid_val = EXP_MID[x_mid];
  Frac128 result = fputil::multiply_add(mid_val, p, mid_val);

  return exp_handle_rounding(result, is_neg, d, e_y, rounding);
}

LIBC_INLINE double exp(double x, [[maybe_unused]] int rounding) {
  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  bool is_neg = xbits.is_neg();
  uint64_t x_val = xbits.uintval();
  uint64_t x_val_abs = xbits.abs().uintval();

  // x < log(2^-1075) or x >= 0x1.6232bdd7abcd3p+9 or |x| < 2^-53.
  if (LIBC_UNLIKELY(x_val >= 0xc087'4910'd52d'3052ULL ||
                    (x_val < 0xbca0'0000'0000'0000ULL &&
                     x_val >= 0x4086'2e42'fefa'39f0ULL) ||
                    x_val < 0x3ca0'0000'0000'0000ULL)) {
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

  uint64_t x_s_shifted = (x_s << 11) | (uint64_t(1) << 63);
  // LSB(x_s_frac) = 2^-52
  Frac64 x_s_frac(x_s_shifted);

  Frac64 x_ln2 = x_s_frac * INV_LN2_F64;
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
  uint64_t x_lo = l2y_r_hi & ((uint64_t(1) << 60) - 1);

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
  constexpr uint32_t ROUNDING_ERROR = 0x800;
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

} // namespace static_rounding

} // namespace math

namespace math {
namespace integer_eval {

LIBC_INLINE double exp(double x) {
  return math::static_rounding::exp(x, FE_TONEAREST);
}

} // namespace integer_eval
} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
