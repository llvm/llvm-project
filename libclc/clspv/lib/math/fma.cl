/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// This version is derived from the generic fma software implementation
// (__clc_sw_fma), but avoids the use of ulong in favor of uint2. The logic has
// been updated as appropriate.

#include <clc/clc.h>
#include "../../../generic/lib/clcmacro.h"
#include "../../../generic/lib/math/math.h"

struct fp {
  uint2 mantissa;
  int exponent;
  uint sign;
};

_CLC_DEF _CLC_OVERLOAD float fma(float a, float b, float c) {
  /* special cases */
  if (isnan(a) || isnan(b) || isnan(c) || isinf(a) || isinf(b)) {
    return mad(a, b, c);
  }

  /* If only c is inf, and both a,b are regular numbers, the result is c*/
  if (isinf(c)) {
    return c;
  }

  a = __clc_flush_denormal_if_not_supported(a);
  b = __clc_flush_denormal_if_not_supported(b);
  c = __clc_flush_denormal_if_not_supported(c);

  if (a == 0.0f || b == 0.0f) {
    return c;
  }

  if (c == 0) {
    return a * b;
  }

  struct fp st_a, st_b, st_c;

  st_a.exponent = a == .0f ? 0 : ((as_uint(a) & 0x7f800000) >> 23) - 127;
  st_b.exponent = b == .0f ? 0 : ((as_uint(b) & 0x7f800000) >> 23) - 127;
  st_c.exponent = c == .0f ? 0 : ((as_uint(c) & 0x7f800000) >> 23) - 127;

  st_a.mantissa.lo = a == .0f ? 0 : (as_uint(a) & 0x7fffff) | 0x800000;
  st_b.mantissa.lo = b == .0f ? 0 : (as_uint(b) & 0x7fffff) | 0x800000;
  st_c.mantissa.lo = c == .0f ? 0 : (as_uint(c) & 0x7fffff) | 0x800000;
  st_a.mantissa.hi = 0;
  st_b.mantissa.hi = 0;
  st_c.mantissa.hi = 0;

  st_a.sign = as_uint(a) & 0x80000000;
  st_b.sign = as_uint(b) & 0x80000000;
  st_c.sign = as_uint(c) & 0x80000000;

  // Multiplication.
  // Move the product to the highest bits to maximize precision
  // mantissa is 24 bits => product is 48 bits, 2bits non-fraction.
  // Add one bit for future addition overflow,
  // add another bit to detect subtraction underflow
  struct fp st_mul;
  st_mul.sign = st_a.sign ^ st_b.sign;
  st_mul.mantissa.hi = mul_hi(st_a.mantissa.lo, st_b.mantissa.lo);
  st_mul.mantissa.lo = st_a.mantissa.lo * st_b.mantissa.lo;
  uint upper_14bits = (st_mul.mantissa.lo >> 18) & 0x3fff;
  st_mul.mantissa.lo <<= 14;
  st_mul.mantissa.hi <<= 14;
  st_mul.mantissa.hi |= upper_14bits;
  st_mul.exponent = (st_mul.mantissa.lo != 0 || st_mul.mantissa.hi != 0)
                        ? st_a.exponent + st_b.exponent
                        : 0;

// Mantissa is 23 fractional bits, shift it the same way as product mantissa
#define C_ADJUST 37ul

  // both exponents are bias adjusted
  int exp_diff = st_mul.exponent - st_c.exponent;

  uint abs_exp_diff = abs(exp_diff);
  st_c.mantissa.hi = (st_c.mantissa.lo << 5);
  st_c.mantissa.lo = 0;
  uint2 cutoff_bits = (uint2)(0, 0);
  uint2 cutoff_mask = (uint2)(0, 0);
  if (abs_exp_diff < 32) {
    cutoff_mask.lo = (1u << abs(exp_diff)) - 1u;
  } else if (abs_exp_diff < 64) {
    cutoff_mask.lo = 0xffffffff;
    uint remaining = abs_exp_diff - 32;
    cutoff_mask.hi = (1u << remaining) - 1u;
  } else {
    cutoff_mask = (uint2)(0, 0);
  }
  uint2 tmp = (exp_diff > 0) ? st_c.mantissa : st_mul.mantissa;
  if (abs_exp_diff > 0) {
    cutoff_bits = abs_exp_diff >= 64 ? tmp : (tmp & cutoff_mask);
    if (abs_exp_diff < 32) {
      // shift some of the hi bits into the shifted lo bits.
      uint shift_mask = (1u << abs_exp_diff) - 1;
      uint upper_saved_bits = tmp.hi & shift_mask;
      upper_saved_bits = upper_saved_bits << (32 - abs_exp_diff);
      tmp.hi >>= abs_exp_diff;
      tmp.lo >>= abs_exp_diff;
      tmp.lo |= upper_saved_bits;
    } else if (abs_exp_diff < 64) {
      tmp.lo = (tmp.hi >> (abs_exp_diff - 32));
      tmp.hi = 0;
    } else {
      tmp = (uint2)(0, 0);
    }
  }
  if (exp_diff > 0)
    st_c.mantissa = tmp;
  else
    st_mul.mantissa = tmp;

  struct fp st_fma;
  st_fma.sign = st_mul.sign;
  st_fma.exponent = max(st_mul.exponent, st_c.exponent);
  st_fma.mantissa = (uint2)(0, 0);
  if (st_c.sign == st_mul.sign) {
    uint carry = (hadd(st_mul.mantissa.lo, st_c.mantissa.lo) >> 31) & 0x1;
    st_fma.mantissa = st_mul.mantissa + st_c.mantissa;
    st_fma.mantissa.hi += carry;
  } else {
    // cutoff bits borrow one
    uint cutoff_borrow = ((cutoff_bits.lo != 0 || cutoff_bits.hi != 0) &&
                          (st_mul.exponent > st_c.exponent))
                             ? 1
                             : 0;
    uint borrow = 0;
    if (st_c.mantissa.lo > st_mul.mantissa.lo) {
      borrow = 1;
    } else if (st_c.mantissa.lo == UINT_MAX && cutoff_borrow == 1) {
      borrow = 1;
    } else if ((st_c.mantissa.lo + cutoff_borrow) > st_mul.mantissa.lo) {
      borrow = 1;
    }

    st_fma.mantissa.lo = st_mul.mantissa.lo - st_c.mantissa.lo - cutoff_borrow;
    st_fma.mantissa.hi = st_mul.mantissa.hi - st_c.mantissa.hi - borrow;
  }

  // underflow: st_c.sign != st_mul.sign, and magnitude switches the sign
  if (st_fma.mantissa.hi > INT_MAX) {
    st_fma.mantissa = ~st_fma.mantissa;
    uint carry = (hadd(st_fma.mantissa.lo, 1u) >> 31) & 0x1;
    st_fma.mantissa.lo += 1;
    st_fma.mantissa.hi += carry;

    st_fma.sign = st_mul.sign ^ 0x80000000;
  }

  // detect overflow/underflow
  uint leading_zeroes = clz(st_fma.mantissa.hi);
  if (leading_zeroes == 32) {
    leading_zeroes += clz(st_fma.mantissa.lo);
  }
  int overflow_bits = 3 - leading_zeroes;

  // adjust exponent
  st_fma.exponent += overflow_bits;

  // handle underflow
  if (overflow_bits < 0) {
    uint shift = -overflow_bits;
    if (shift < 32) {
      uint shift_mask = (1u << shift) - 1;
      uint saved_lo_bits = (st_fma.mantissa.lo >> (32 - shift)) & shift_mask;
      st_fma.mantissa.lo <<= shift;
      st_fma.mantissa.hi <<= shift;
      st_fma.mantissa.hi |= saved_lo_bits;
    } else if (shift < 64) {
      st_fma.mantissa.hi = (st_fma.mantissa.lo << (64 - shift));
      st_fma.mantissa.lo = 0;
    } else {
      st_fma.mantissa = (uint2)(0, 0);
    }

    overflow_bits = 0;
  }

  // rounding
  // overflow_bits is now in the range of [0, 3] making the shift greater than
  // 32 bits.
  uint2 trunc_mask;
  uint trunc_shift = C_ADJUST + overflow_bits - 32;
  trunc_mask.hi = (1u << trunc_shift) - 1;
  trunc_mask.lo = UINT_MAX;
  uint2 trunc_bits = st_fma.mantissa & trunc_mask;
  trunc_bits.lo |= (cutoff_bits.hi != 0 || cutoff_bits.lo != 0) ? 1 : 0;
  uint2 last_bit;
  last_bit.lo = 0;
  last_bit.hi = st_fma.mantissa.hi & (1u << trunc_shift);
  uint grs_shift = C_ADJUST - 3 + overflow_bits - 32;
  uint2 grs_bits;
  grs_bits.lo = 0;
  grs_bits.hi = 0x4u << grs_shift;

  // round to nearest even
  if ((trunc_bits.hi > grs_bits.hi ||
       (trunc_bits.hi == grs_bits.hi && trunc_bits.lo > grs_bits.lo)) ||
      (trunc_bits.hi == grs_bits.hi && trunc_bits.lo == grs_bits.lo &&
       last_bit.hi != 0)) {
    uint shift = C_ADJUST + overflow_bits - 32;
    st_fma.mantissa.hi += 1u << shift;
  }

        // Shift mantissa back to bit 23
  st_fma.mantissa.lo = (st_fma.mantissa.hi >> (C_ADJUST + overflow_bits - 32));
  st_fma.mantissa.hi = 0;

  // Detect rounding overflow
  if (st_fma.mantissa.lo > 0xffffff) {
    ++st_fma.exponent;
    st_fma.mantissa.lo >>= 1;
  }

  if (st_fma.mantissa.lo == 0) {
    return 0.0f;
  }

  // Flating point range limit
  if (st_fma.exponent > 127) {
    return as_float(as_uint(INFINITY) | st_fma.sign);
  }

  // Flush denormals
  if (st_fma.exponent <= -127) {
    return as_float(st_fma.sign);
  }

  return as_float(st_fma.sign | ((st_fma.exponent + 127) << 23) |
                  ((uint)st_fma.mantissa.lo & 0x7fffff));
}
_CLC_TERNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, fma, float, float, float)
