//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains an AdvSIMD architecture optimised single-precision expf.
///
//===----------------------------------------------------------------------===//

#include "src/mathvec/expf.h"
#include "arm_neon.h"
#include "src/__support/common.h"
#include "src/__support/mathvec/expf_utils.h"
#include "src/mathvec/aarch64/common.h"

namespace LIBC_NAMESPACE_DECL {

struct expf_data {
  float64x2_t shift, inv_ln2;
  double ln2_hi, ln2_lo;
  double c1, c3;
  float64x2_t c0, c2;
  float32x4_t range_val;
  uint32x4_t inf;
  uint64x2_t idx_mask;
  const uint64_t *mantissa;
};

static constexpr expf_data EXPF_DATA = {
    V2(0x1.800000000ffc0p+46), // shift
    V2(0x1.71547652b82fep+0),  // inv_ln2
    0x1.62e42fefa39efp-1,      // ln2_hi
    0x1.abc9e3b39803fp-56,     // ln2_lo
    0x1.55555555543c2p-3,      // c1
    0x1.111126b4eff73p-7,      // c3
    V2(0x1.fffffffffdbcep-2),  // c0
    V2(0x1.555573c64f2e3p-5),  // c2
    V4(0x1p+9),                // range_val
    V4(0x7f800000),            // inf
    V2(0x3f),                  // idx_mask
    mathvec::EXP_MANTISSA,     // mantissa
};

LIBC_INLINE static float64x2_t exp_lookup(uint64x2_t u, const expf_data &data) {
  // The low 6 bits of u index the 64 element mantissa table.
  uint64_t idx0 = vgetq_lane_u64(u & data.idx_mask, 0);
  uint64_t idx1 = vgetq_lane_u64(u & data.idx_mask, 1);

  uint64_t mant0 = data.mantissa[idx0];
  uint64_t mant1 = data.mantissa[idx1];

  uint64x2_t mantissa = vdupq_n_u64(mant0);
  mantissa = vsetq_lane_u64(mant1, mantissa, 1);

  // The next 11 bits, 16:6, holds the biased exponent for the result.
  // Shifting by 46 moves these bits up to the exponent field.
  uint64x2_t exponent = vshlq_n_u64(u, 46);
  uint64x2_t mask = vdupq_n_u64(0xfff0000000000000);
  uint64x2_t result = vbslq_u64(mask, exponent, mantissa);

  return vreinterpretq_f64_u64(result);
}

LIBC_INLINE static float64x2_t inline_exp(float64x2_t x,
                                          const expf_data &data) {
  float64x2_t z = vfmaq_f64(data.shift, x, data.inv_ln2);
  float64x2_t n = vsubq_f64(z, data.shift);

  float64x2_t ln2 = vld1q_f64(&data.ln2_hi);

  float64x2_t r = x;
  r = vfmsq_laneq_f64(r, n, ln2, 0);
  r = vfmsq_laneq_f64(r, n, ln2, 1);

  float64x2_t coeffs = vld1q_f64(&data.c1);

  // poly(r) = exp(r) - 1 ~= r + c0*r^2 + c1*r^3 + c2*r^4 + c3*r^5
  float64x2_t r2 = r * r;
  float64x2_t p01 = vfmaq_laneq_f64(data.c0, r, coeffs, 0);
  float64x2_t p23 = vfmaq_laneq_f64(data.c2, r, coeffs, 1);
  float64x2_t p04 = vfmaq_f64(p01, r2, p23);
  float64x2_t y = vfmaq_f64(r, r2, p04);

  uint64x2_t u = vreinterpretq_u64_f64(z);
  float64x2_t s = exp_lookup(u, data);

  return vfmaq_f64(s, s, y);
}

LLVM_LIBC_FUNCTION(AdvSIMDFP32Vector, expf, (AdvSIMDFP32Vector x),
                   "_ZGVnN4v_expf") {
  const expf_data &data = *PTR_BARRIER(&EXPF_DATA);

  // Splits into an upper and lower half for double-precision computation.
  float64x2_t x_d_lo = vcvt_f64_f32(vget_low_f32(x));
  float64x2_t x_d_hi = vcvt_high_f64_f32(x);

  // Compute the double precision exponential for the high and low halves.
  float64x2_t y_lo = inline_exp(x_d_lo, data);
  float64x2_t y_hi = inline_exp(x_d_hi, data);

  // Round to single precision, and recombine the results.
  float32x4_t ret = vcombine_f32(vcvt_f32_f64(y_lo), vcvt_f32_f64(y_hi));

  // Handle special cases for overflow and underflow.
  uint32x4_t special = vcagtq_f32(x, data.range_val);
  bool has_special = vmaxvq_u32(special) != 0;
  if (LIBC_UNLIKELY(has_special)) {
    uint32x4_t is_inf = vcgtzq_f32(x);
    uint32x4_t inf_or_zero = vandq_u32(is_inf, data.inf);
    float32x4_t special_res = vreinterpretq_f32_u32(inf_or_zero);

    // Combine the results for normal and special cases and return.
    return vbslq_f32(special, special_res, ret);
  }

  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
