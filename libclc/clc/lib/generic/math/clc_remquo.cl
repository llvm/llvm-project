//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/integer/clc_clz.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_floor.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_ldexp.h>
#include <clc/math/clc_rint.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/clc_trunc.h>
#include <clc/math/math.h>
#include <clc/shared/clc_max.h>

#define _sHighMask 0xfffff000u
#define _iMaxQExp 0xbu
// To prevent YLow to be denormal it should be checked
// that Exp(Y) <= -127+23 (worst case when only last bit is non zero)
//      Exp(Y) < -103 -> Y < 0x0C000000
// That value is used to construct _iYSub by setting up first bit to 1.
// _iYCmp is get from max acceptable value 0x797fffff:
//   0x797fffff - 0x8C000000 = 0x(1)ED7FFFFF
#define _iYSub 0x8C000000u
#define _iYCmp 0xED7FFFFFu
#define _iOne 0x00000001u

static _CLC_INLINE int internal_remquo(float x, float y, private float *r,
                                       private uint *q) {
  uint signif_x, signif_y, rem_bit, quo_bit, tmp_x, tmp_y;
  int exp_x, exp_y, i, j;
  uint expabs_diff, special_op = 0, signbit_x, signbit_y, sign = 1;
  float result, abs_x, abs_y;
  float zero = 0.0f;
  int nRet = 0;
  // Remove sign bits
  tmp_x = ((*(int *)&x)) & EXSIGNBIT_SP32;
  tmp_y = ((*(int *)&y)) & EXSIGNBIT_SP32;
  signbit_x = (uint)((*(int *)&x) >> 31);
  signbit_y = (uint)((*(int *)&y) >> 31);
  if (signbit_x ^ signbit_y)
    sign = (-sign);
  // Get float absolute values
  abs_x = *(float *)&tmp_x;
  abs_y = *(float *)&tmp_y;
  // Remove exponent bias
  exp_x = (int)((tmp_x & (0x7F800000)) >> 23) - 127;
  exp_y = (int)((tmp_y & (0x7F800000)) >> 23) - 127;
  // Test for NaNs, Infs, and Zeros
  if ((exp_x == 0x00000080) || (exp_y == 0x00000080) || (tmp_x == 0) ||
      (tmp_y == 0))
    special_op++;
  // Get significands
  signif_x = (tmp_x & MANTBITS_SP32);
  signif_y = (tmp_y & MANTBITS_SP32);
  // Process NaNs, Infs, and Zeros
  if (special_op) {
    (*q) = 0;
    // x is NaN
    if ((signif_x != 0) && (exp_x == 0x00000080))
      result = x * 1.7f;
    // y is NaN
    else if ((signif_y != 0) && (exp_y == 0x00000080))
      result = y * 1.7f;
    // y is zero
    else if (abs_y == zero) {
      result = zero / zero;
      nRet = 1;
    }
    // x is zero
    else if (abs_x == zero)
      result = x;
    // x is Inf
    else if ((signif_x == 0) && (exp_x == 0x00000080))
      result = zero / zero;
    // y is Inf
    else
      result = x;
    (*r) = (result);
    return nRet;
  }
  // If x < y, fast return
  if (abs_x <= abs_y) {
    (*q) = 1 * sign;
    if (abs_x == abs_y) {
      (*r) = (zero * x);
      return nRet;
    }
    // Is x too big to scale up by 2.0f?
    if (exp_x != 127) {
      if ((2.0f * abs_x) <= abs_y) {
        (*q) = 0;
        (*r) = x;
        return nRet;
      }
    }
    result = abs_x - abs_y;
    if (signbit_x) {
      result = -result;
    }
    (*r) = (result);
    return nRet;
  }
  // Check for denormal x and y, adjust and normalize
  if ((exp_x == -127) && (signif_x != 0)) {
    exp_x = -126;
    while (signif_x <= MANTBITS_SP32) {
      exp_x--;
      signif_x <<= 1;
    };
  } else
    signif_x = (signif_x | (0x00800000L));
  if ((exp_y == -127) && (signif_y != 0)) {
    exp_y = -126;
    while (signif_y <= MANTBITS_SP32) {
      exp_y--;
      signif_y <<= 1;
    };
  } else
    signif_y = (signif_y | (0x00800000L));
  //
  // Main computational path
  //
  // Calculate exponent difference
  expabs_diff = (exp_x - exp_y) + 1;
  rem_bit = signif_x;
  quo_bit = 0;
  for (i = 0; i < expabs_diff; i++) {
    quo_bit = quo_bit << 1;
    if (rem_bit >= signif_y) {
      rem_bit -= signif_y;
      quo_bit++;
    }
    rem_bit <<= 1;
  }
  // Zero remquo ... return immediately with sign of x
  if (rem_bit == 0) {
    (*q) = ((uint)(0x7FFFFFFFL & quo_bit)) * sign;
    (*r) = (zero * x);
    return nRet;
  }
  // Adjust remquo
  rem_bit >>= 1;
  // Set exponent base, unbiased
  j = exp_y;
  // Calculate normalization shift
  while (rem_bit <= MANTBITS_SP32) {
    j--;
    rem_bit <<= 1;
  };
  // Prepare normal results
  if (j >= -126) {
    // Remove explicit 1
    rem_bit &= MANTBITS_SP32;
    // Set final exponent ... add exponent bias
    j = j + 127;
  }
  // Prepare denormal results
  else {
    // Determine denormalization shift count
    j = -j - 126;
    // Denormalization
    rem_bit >>= j;
    // Set final exponent ... denorms are 0
    j = 0;
  }
  rem_bit = (((uint)(j)) << 23) | rem_bit;
  // Create float result and adjust if >= .5 * divisor
  result = *(float *)&rem_bit;
  if ((2.0f * result) >= abs_y) {
    if ((2.0f * result) == abs_y) {
      if (quo_bit & 0x01) {
        result = -result;
        quo_bit++;
      }
    } else {
      result = result - abs_y;
      quo_bit++;
    }
  }
  // Final adjust for sign of input
  (*q) = ((uint)(0x7FFFFFFF & (quo_bit))) * sign;
  if (signbit_x)
    result = -result;
  (*r) = (result);
  return nRet;
}

#define __CLC_ADDRESS_SPACE private
#include <clc_remquo.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_ADDRESS_SPACE global
#include <clc_remquo.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_ADDRESS_SPACE local
#include <clc_remquo.inc>
#undef __CLC_ADDRESS_SPACE

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_ADDRESS_SPACE generic
#include <clc_remquo.inc>
#undef __CLC_ADDRESS_SPACE
#endif
