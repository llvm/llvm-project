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

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/integer/clc_abs.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_isnan.h>
#include <clc/shared/clc_max.h>

#include "config.h"
#include "math.h"

struct fp {
  ulong mantissa;
  int exponent;
  uint sign;
};

_CLC_DEF _CLC_OVERLOAD float __clc_sw_fma(float a, float b, float c) {
  /* special cases */
  if (__clc_isnan(a) || __clc_isnan(b) || __clc_isnan(c) || __clc_isinf(a) ||
      __clc_isinf(b))
    return mad(a, b, c);

  /* If only c is inf, and both a,b are regular numbers, the result is c*/
  if (__clc_isinf(c))
    return c;

  a = __clc_flush_denormal_if_not_supported(a);
  b = __clc_flush_denormal_if_not_supported(b);
  c = __clc_flush_denormal_if_not_supported(c);

  if (c == 0)
    return a * b;

  struct fp st_a, st_b, st_c;

  st_a.exponent = a == .0f ? 0 : ((as_uint(a) & 0x7f800000) >> 23) - 127;
  st_b.exponent = b == .0f ? 0 : ((as_uint(b) & 0x7f800000) >> 23) - 127;
  st_c.exponent = c == .0f ? 0 : ((as_uint(c) & 0x7f800000) >> 23) - 127;

  st_a.mantissa = a == .0f ? 0 : (as_uint(a) & 0x7fffff) | 0x800000;
  st_b.mantissa = b == .0f ? 0 : (as_uint(b) & 0x7fffff) | 0x800000;
  st_c.mantissa = c == .0f ? 0 : (as_uint(c) & 0x7fffff) | 0x800000;

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
  st_mul.mantissa = (st_a.mantissa * st_b.mantissa) << 14ul;
  st_mul.exponent = st_mul.mantissa ? st_a.exponent + st_b.exponent : 0;

  // FIXME: Detecting a == 0 || b == 0 above crashed GCN isel
  if (st_mul.exponent == 0 && st_mul.mantissa == 0)
    return c;

// Mantissa is 23 fractional bits, shift it the same way as product mantissa
#define C_ADJUST 37ul

  // both exponents are bias adjusted
  int exp_diff = st_mul.exponent - st_c.exponent;

  st_c.mantissa <<= C_ADJUST;
  ulong cutoff_bits = 0;
  ulong cutoff_mask = (1ul << __clc_abs(exp_diff)) - 1ul;
  if (exp_diff > 0) {
    cutoff_bits =
        exp_diff >= 64 ? st_c.mantissa : (st_c.mantissa & cutoff_mask);
    st_c.mantissa = exp_diff >= 64 ? 0 : (st_c.mantissa >> exp_diff);
  } else {
    cutoff_bits =
        -exp_diff >= 64 ? st_mul.mantissa : (st_mul.mantissa & cutoff_mask);
    st_mul.mantissa = -exp_diff >= 64 ? 0 : (st_mul.mantissa >> -exp_diff);
  }

  struct fp st_fma;
  st_fma.sign = st_mul.sign;
  st_fma.exponent = __clc_max(st_mul.exponent, st_c.exponent);
  if (st_c.sign == st_mul.sign) {
    st_fma.mantissa = st_mul.mantissa + st_c.mantissa;
  } else {
    // cutoff bits borrow one
    st_fma.mantissa =
        st_mul.mantissa - st_c.mantissa -
        (cutoff_bits && (st_mul.exponent > st_c.exponent) ? 1 : 0);
  }

  // underflow: st_c.sign != st_mul.sign, and magnitude switches the sign
  if (st_fma.mantissa > LONG_MAX) {
    st_fma.mantissa = 0 - st_fma.mantissa;
    st_fma.sign = st_mul.sign ^ 0x80000000;
  }

  // detect overflow/underflow
  int overflow_bits = 3 - clz(st_fma.mantissa);

  // adjust exponent
  st_fma.exponent += overflow_bits;

  // handle underflow
  if (overflow_bits < 0) {
    st_fma.mantissa <<= -overflow_bits;
    overflow_bits = 0;
  }

  // rounding
  ulong trunc_mask = (1ul << (C_ADJUST + overflow_bits)) - 1;
  ulong trunc_bits = (st_fma.mantissa & trunc_mask) | (cutoff_bits != 0);
  ulong last_bit = st_fma.mantissa & (1ul << (C_ADJUST + overflow_bits));
  ulong grs_bits = (0x4ul << (C_ADJUST - 3 + overflow_bits));

  // round to nearest even
  if ((trunc_bits > grs_bits) || (trunc_bits == grs_bits && last_bit != 0))
    st_fma.mantissa += (1ul << (C_ADJUST + overflow_bits));

  // Shift mantissa back to bit 23
  st_fma.mantissa = (st_fma.mantissa >> (C_ADJUST + overflow_bits));

  // Detect rounding overflow
  if (st_fma.mantissa > 0xffffff) {
    ++st_fma.exponent;
    st_fma.mantissa >>= 1;
  }

  if (st_fma.mantissa == 0)
    return .0f;

  // Flating point range limit
  if (st_fma.exponent > 127)
    return as_float(as_uint(INFINITY) | st_fma.sign);

  // Flush denormals
  if (st_fma.exponent <= -127)
    return as_float(st_fma.sign);

  return as_float(st_fma.sign | ((st_fma.exponent + 127) << 23) |
                  ((uint)st_fma.mantissa & 0x7fffff));
}
_CLC_TERNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_sw_fma, float,
                       float, float)
