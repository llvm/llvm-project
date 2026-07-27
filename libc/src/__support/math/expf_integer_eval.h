//===-- Implementation header for expf using integer-only --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/frac32.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/check/exp_exceptions.h"
#include "src/__support/math/exp_integer_utils.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace integer_only {

// Round-nearest, no except implementation of expf using integer-only
// arithmetic.
// LIBC_INLINE float expf(float x) {
//   using FPBits = typename fputil::FPBits<float>;
//   FPBits xbits(x);

//   bool is_neg = xbits.is_neg();
//   uint32_t x_val = xbits.uintval();
//   uint32_t x_val_abs = x_val & 0x7fff'ffffU;

//   // When |x| >= 89, |x| < 2^-25, or x is NaN
//   if (LIBC_UNLIKELY(x_val_abs >= 0x42b2'0000U || x_val_abs <= 0x3380'0000U))
//   {
//     if (x_val_abs < 0x3300'0000U) { // |x| < 2^-25
//       return 1.0f;
//     }

//     if (x_val_abs < 0x3380'0000U) { // |x| < 2^-24
//       return 1.0f + x;
//     }

//     if (xbits.is_nan()) {
//       // Per conversation with lntue, we don't need to raise exception here,
//       // as we're assuming no FPUs/fenv in this kind of environment
//       if (xbits.is_signaling_nan()) {
//         // silencing
//         return FPBits::quiet_nan().get_val();
//       }

//       // quiet NaN
//       return x;
//     }

//     // e^-inf = 0
//     // e^+inf = +inf
//     if (xbits.is_inf()) {
//       return is_neg ? 0.0f : FPBits::inf().get_val();
//     }

//     // Large finite positive --> overflow
//     if (!is_neg) {
//       return FPBits::inf().get_val();
//     }

//     // x < log(2^-150) or NaN (NaN is already handled above)
//     if (xbits.uintval() >= 0xc2cf'f1b5U) {
//       return 0.0f;
//     }
//   }

//   if (LIBC_UNLIKELY(x_val >= check::exp_internal::Bounds<float>::UPPER_BITS
//   &&
//                     !is_neg)) { // overflow
//     return FPBits::inf().get_val();
//   }

//   // Main calculations

//   uint16_t x_e = xbits.get_biased_exponent();
//   uint64_t x_u = xbits.get_mantissa();

//   // Range reduction
//   // The algorithm near the end of this function estimates 2^r,
//   // where r is the fractional part of x * log2(e) and is in [0, 1].
//   // See EXPF_COEFFS for more details on the approximation polynomial used.

//   // add leading bit = 1
//   x_u |= uint64_t(1) << FPBits::FRACTION_LEN;

//   // shift to top 32 bit --> decimal point at hidden bit that we've added
//   x_u <<= 32;

//   int x_e_unbiased = static_cast<int>(x_e) - FPBits::EXP_BIAS;

//   // shift for the decimal point to be at the hidden bit
//   if (x_e_unbiased > 0) {
//     x_u <<= x_e_unbiased;
//   } else if (x_e_unbiased < 0) {
//     x_u >>= -x_e_unbiased;
//   }

//   // LSB(x_u_frac) = 2^-55
//   Frac64 x_u_frac(x_u);

//   // LSB(x_ln2) = 2^-54
//   Frac64 x_ln2 = x_u_frac * INV_LN2;

//   constexpr uint64_t FRAC_MASK = (uint64_t(1) << 54) - 1;
//   uint64_t x_ln2_bit = x_ln2.val[0];

//   uint64_t e_y, l2y_r;
//   uint32_t e_y_unbiased;

//   if (LIBC_UNLIKELY(is_neg)) {
//     e_y = (x_ln2_bit + FRAC_MASK) & ~FRAC_MASK;
//     l2y_r = e_y - x_ln2_bit;
//     e_y_unbiased = (FPBits::EXP_BIAS << 23) - static_cast<uint32_t>(e_y >>
//     31);
//   } else {
//     e_y = x_ln2_bit & ~FRAC_MASK;
//     l2y_r = x_ln2_bit - e_y;
//     e_y_unbiased = (FPBits::EXP_BIAS << 23) + static_cast<uint32_t>(e_y >>
//     31);
//   }

//   // LSB(l2y_r_frac) = LSB(l2y_r) * 2^-10 = 2^-64
//   Frac64 l2y_r_frac(l2y_r << 10);

//   // p = 2^l2y_r_frac - 1
//   Frac64 p = fputil::polyeval(l2y_r_frac, Frac64(0), EXPF_COEFFS[0],
//                               EXPF_COEFFS[1], EXPF_COEFFS[2], EXPF_COEFFS[3],
//                               EXPF_COEFFS[4], EXPF_COEFFS[5]);

//   uint32_t k = static_cast<uint32_t>(e_y >> 54);
//   int d = static_cast<int>(k) - FPBits::EXP_BIAS;

//   if (LIBC_UNLIKELY(is_neg && d >= 23)) { // underflow
//     return 0.0f;
//   }

//   if (LIBC_UNLIKELY(is_neg && d >= 0)) { // subnormal
//     uint64_t full_val = (uint64_t(1) << 63) | (p.val[0] >> 1);

//     // add rounding bit
//     full_val += (uint64_t(1) << (40 + d));

//     // shift back to align to 32-bit float representation
//     uint32_t result = static_cast<uint32_t>(full_val >> (41 + d));

//     return cpp::bit_cast<float>(result);
//   }

//   // RN 23 bits --> shift for the LSB to be 2^-24 --> +1, shift for another
//   // bit
//   // Can be rounded up
//   // + e_y_unbiased

//   uint32_t result = (static_cast<uint32_t>(p.val[0] >> 40) + 1);
//   result >>= 1;

//   result += e_y_unbiased;

//   return cpp::bit_cast<float>(result);
// }

// Round-nearest, no except implementation of expf using integer-only
// arithmetic.
// Frac32 implementation
LIBC_INLINE float expf(float x) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);

  bool is_neg = xbits.is_neg();
  uint32_t x_val = xbits.uintval();
  uint32_t x_val_abs = x_val & 0x7fff'ffffU;

  // When |x| >= 89, |x| < 2^-25, or x is NaN
  if (LIBC_UNLIKELY(x_val_abs >= 0x42b2'0000U || x_val_abs <= 0x3380'0000U)) {
    if (x_val_abs < 0x3300'0000U) { // |x| < 2^-25
      return 1.0f;
    }

    if (x_val_abs < 0x3380'0000U) { // |x| < 2^-24
      return 1.0f + x;
    }

    if (xbits.is_nan()) {
      // Per conversation with lntue, we don't need to raise exception here,
      // as we're assuming no FPUs/fenv in this kind of environment
      if (xbits.is_signaling_nan()) {
        // silencing
        return FPBits::quiet_nan().get_val();
      }

      // quiet NaN
      return x;
    }

    // e^-inf = 0
    // e^+inf = +inf
    if (xbits.is_inf()) {
      return is_neg ? 0.0f : FPBits::inf().get_val();
    }

    // Large finite positive --> overflow
    if (!is_neg) {
      return FPBits::inf().get_val();
    }

    // x < log(2^-150) or NaN (NaN is already handled above)
    if (xbits.uintval() >= 0xc2cf'f1b5U) {
      return 0.0f;
    }
  }

  if (LIBC_UNLIKELY(x_val >= check::exp_internal::Bounds<float>::UPPER_BITS &&
                    !is_neg)) { // overflow
    return FPBits::inf().get_val();
  }

  // Main calculations

  uint16_t x_e = xbits.get_biased_exponent();
  uint32_t x_u = xbits.get_mantissa();

  // Range reduction
  // The algorithm near the end of this function estimates 2^r,
  // where r is the fractional part of x * log2(e) and is in [0, 1].
  // See EXPF_COEFFS_FRAC32 for more details on the approximation polynomial
  // used.

  // add leading bit = 1
  x_u |= uint32_t(1) << FPBits::FRACTION_LEN;

  int x_e_unbiased = static_cast<int>(x_e) - FPBits::EXP_BIAS;

  // shift for the decimal point to be at the hidden bit
  if (x_e_unbiased > 0) {
    x_u <<= x_e_unbiased;
  } else if (x_e_unbiased < 0) {
    x_u >>= -x_e_unbiased;
  }

  // LSB(x_u_frac) = 2^-23
  Frac32 x_u_frac(x_u);

  // LSB(x_ln2) = 2^-22
  Frac32 x_ln2 = x_u_frac * INV_LN2_FRAC32;

  constexpr uint32_t FRAC_MASK = (uint32_t(1) << 22) - 1;
  uint32_t x_ln2_bit = x_ln2.val[0];

  uint32_t e_y, l2y_r;
  uint32_t e_y_unbiased;

  if (LIBC_UNLIKELY(is_neg)) {
    e_y = (x_ln2_bit + FRAC_MASK) & ~FRAC_MASK;
    l2y_r = e_y - x_ln2_bit;
    e_y_unbiased = (FPBits::EXP_BIAS << 23) - static_cast<uint32_t>(e_y << 1);
  } else {
    e_y = x_ln2_bit & ~FRAC_MASK;
    l2y_r = x_ln2_bit - e_y;
    e_y_unbiased = (FPBits::EXP_BIAS << 23) + static_cast<uint32_t>(e_y << 1);
  }

  // LSB(l2y_r_frac) = LSB(l2y_r) * 2^-10 = 2^-32
  Frac32 l2y_r_frac(l2y_r << 10);

  // p = 2^l2y_r_frac - 1
  Frac32 p = fputil::polyeval(l2y_r_frac, Frac32(0), EXPF_COEFFS_FRAC32[0],
                              EXPF_COEFFS_FRAC32[1], EXPF_COEFFS_FRAC32[2],
                              EXPF_COEFFS_FRAC32[3], EXPF_COEFFS_FRAC32[4],
                              EXPF_COEFFS_FRAC32[5]);

  uint32_t k = static_cast<uint32_t>(e_y >> 22);
  int d = static_cast<int>(k) - FPBits::EXP_BIAS;

  if (LIBC_UNLIKELY(is_neg && d >= 23)) { // underflow
    return 0.0f;
  }

  if (LIBC_UNLIKELY(is_neg && d >= 0)) { // subnormal
    // 1 + p
    uint64_t full_val = (uint64_t(1) << 32) | p.val[0];

    // add rounding bit
    full_val += (uint64_t(1) << (9 + d));

    // shift back to align to 32-bit float representation
    uint32_t result = static_cast<uint32_t>(full_val >> (10 + d));

    return cpp::bit_cast<float>(result);
  }

  // RN 23 bits --> shift for the LSB to be 2^-24 --> +1, shift for another bit
  // Can be rounded up
  // + e_y_unbiased

  uint32_t result = (static_cast<uint32_t>(p.val[0] >> 8) + 1);
  result >>= 1;

  result += e_y_unbiased;

  return cpp::bit_cast<float>(result);
}

} // namespace integer_only

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
