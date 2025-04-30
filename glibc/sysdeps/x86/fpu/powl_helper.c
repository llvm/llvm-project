/* Implement powl for x86 using extra-precision log.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <stdbool.h>

/* High parts and low parts of -log (k/16), for integer k from 12 to
   24.  */

static const long double powl_log_table[] =
  {
    0x4.9a58844d36e49e1p-4L, -0x1.0522624fd558f574p-68L,
    0x3.527da7915b3c6de4p-4L, 0x1.7d4ef4b901b99b9ep-68L,
    0x2.22f1d044fc8f7bc8p-4L, -0x1.8e97c071a42fc388p-68L,
    0x1.08598b59e3a0688ap-4L, 0x3.fd9bf503372c12fcp-72L,
    -0x0p+0L, 0x0p+0L,
    -0xf.85186008b15330cp-8L, 0x1.9b47488a6687672cp-72L,
    -0x1.e27076e2af2e5e9ep-4L, -0xa.87ffe1fe9e155dcp-72L,
    -0x2.bfe60e14f27a791p-4L, 0x1.83bebf1bdb88a032p-68L,
    -0x3.91fef8f353443584p-4L, -0xb.b03de5ff734495cp-72L,
    -0x4.59d72aeae98380e8p-4L, 0xc.e0aa3be4747dc1p-72L,
    -0x5.1862f08717b09f4p-4L, -0x2.decdeccf1cd10578p-68L,
    -0x5.ce75fdaef401a738p-4L, -0x9.314feb4fbde5aaep-72L,
    -0x6.7cc8fb2fe612fcbp-4L, 0x2.5ca2642feb779f98p-68L,
  };

/* High 32 bits of log2 (e), and remainder rounded to 64 bits.  */
static const long double log2e_hi = 0x1.71547652p+0L;
static const long double log2e_lo = 0xb.82fe1777d0ffda1p-36L;

/* Given a number with high part HI and low part LO, add the number X
   to it and store the result in *RHI and *RLO.  It is given that
   either |X| < |0.7 * HI|, or HI == LO == 0, and that the values are
   small enough that no overflow occurs.  The result does not need to
   be exact to 128 bits; 78-bit accuracy of the final accumulated
   result suffices.  */

static inline void
acc_split (long double *rhi, long double *rlo, long double hi, long double lo,
	   long double x)
{
  long double thi = hi + x;
  long double tlo = (hi - thi) + x + lo;
  *rhi = thi + tlo;
  *rlo = (thi - *rhi) + tlo;
}

extern long double __powl_helper (long double x, long double y);
libm_hidden_proto (__powl_helper)

/* Given X a value that is finite and nonzero, or a NaN, and Y a
   finite nonzero value with 0x1p-79 <= |Y| <= 0x1p78, compute X to
   the power Y.  */

long double
__powl_helper (long double x, long double y)
{
  if (isnan (x))
    return __ieee754_expl (y * __ieee754_logl (x));
  bool negate;
  if (x < 0)
    {
      long double absy = fabsl (y);
      if (absy >= 0x1p64L)
	negate = false;
      else
	{
	  unsigned long long yll = absy;
	  if (yll != absy)
	    return __ieee754_expl (y * __ieee754_logl (x));
	  negate = (yll & 1) != 0;
	}
      x = fabsl (x);
    }
  else
    negate = false;

  /* We need to compute Y * log2 (X) to at least 64 bits after the
     point for normal results (that is, to at least 78 bits
     precision).  */
  int x_int_exponent;
  long double x_frac;
  x_frac = __frexpl (x, &x_int_exponent);
  if (x_frac <= 0x0.aaaaaaaaaaaaaaaap0L) /* 2.0L / 3.0L, rounded down */
    {
      x_frac *= 2.0;
      x_int_exponent--;
    }

  long double log_x_frac_hi, log_x_frac_lo;
  /* Determine an initial approximation to log (X_FRAC) using
     POWL_LOG_TABLE, and multiply by a value K/16 to reduce to an
     interval (24/25, 26/25).  */
  int k = (int) ((16.0L / x_frac) + 0.5L);
  log_x_frac_hi = powl_log_table[2 * k - 24];
  log_x_frac_lo = powl_log_table[2 * k - 23];
  long double x_frac_low;
  if (k == 16)
    x_frac_low = 0.0L;
  else
    {
      /* Mask off low 5 bits of X_FRAC so the multiplication by K/16
	 is exact.  These bits are small enough that they can be
	 corrected for by adding log2 (e) * X_FRAC_LOW to the final
	 result.  */
      int32_t se;
      uint32_t i0, i1;
      GET_LDOUBLE_WORDS (se, i0, i1, x_frac);
      x_frac_low = x_frac;
      i1 &= 0xffffffe0;
      SET_LDOUBLE_WORDS (x_frac, se, i0, i1);
      x_frac_low -= x_frac;
      x_frac_low /= x_frac;
      x_frac *= k / 16.0L;
    }

  /* Now compute log (X_FRAC) for X_FRAC in (24/25, 26/25).  Separate
     W = X_FRAC - 1 into high 16 bits and remaining bits, so that
     multiplications for low-order power series terms are exact.  The
     remaining bits are small enough that adding a 64-bit value of
     log2 (1 + W_LO / (1 + W_HI)) will be a sufficient correction for
     them.  */
  long double w = x_frac - 1;
  long double w_hi, w_lo;
  int32_t se;
  uint32_t i0, i1;
  GET_LDOUBLE_WORDS (se, i0, i1, w);
  i0 &= 0xffff0000;
  i1 = 0;
  SET_LDOUBLE_WORDS (w_hi, se, i0, i1);
  w_lo = w - w_hi;
  long double wp = w_hi;
  acc_split (&log_x_frac_hi, &log_x_frac_lo, log_x_frac_hi, log_x_frac_lo, wp);
  wp *= -w_hi;
  acc_split (&log_x_frac_hi, &log_x_frac_lo, log_x_frac_hi, log_x_frac_lo,
	     wp / 2.0L);
  wp *= -w_hi;
  acc_split (&log_x_frac_hi, &log_x_frac_lo, log_x_frac_hi, log_x_frac_lo,
	     wp * 0x0.5555p0L); /* -W_HI**3 / 3, high part.  */
  acc_split (&log_x_frac_hi, &log_x_frac_lo, log_x_frac_hi, log_x_frac_lo,
	     wp * 0x0.5555555555555555p-16L); /* -W_HI**3 / 3, low part.  */
  wp *= -w_hi;
  acc_split (&log_x_frac_hi, &log_x_frac_lo, log_x_frac_hi, log_x_frac_lo,
	     wp / 4.0L);
  /* Subsequent terms are small enough that they only need be computed
     to 64 bits.  */
  for (int i = 5; i <= 17; i++)
    {
      wp *= -w_hi;
      acc_split (&log_x_frac_hi, &log_x_frac_lo, log_x_frac_hi, log_x_frac_lo,
		 wp / i);
    }

  /* Convert LOG_X_FRAC_HI + LOG_X_FRAC_LO to a base-2 logarithm.  */
  long double log2_x_frac_hi, log2_x_frac_lo;
  long double log_x_frac_hi32, log_x_frac_lo64;
  GET_LDOUBLE_WORDS (se, i0, i1, log_x_frac_hi);
  i1 = 0;
  SET_LDOUBLE_WORDS (log_x_frac_hi32, se, i0, i1);
  log_x_frac_lo64 = (log_x_frac_hi - log_x_frac_hi32) + log_x_frac_lo;
  long double log2_x_frac_hi1 = log_x_frac_hi32 * log2e_hi;
  long double log2_x_frac_lo1
    = log_x_frac_lo64 * log2e_hi + log_x_frac_hi * log2e_lo;
  log2_x_frac_hi = log2_x_frac_hi1 + log2_x_frac_lo1;
  log2_x_frac_lo = (log2_x_frac_hi1 - log2_x_frac_hi) + log2_x_frac_lo1;

  /* Correct for the masking off of W_LO.  */
  long double log2_1p_w_lo;
  asm ("fyl2xp1"
       : "=t" (log2_1p_w_lo)
       : "0" (w_lo / (1.0L + w_hi)), "u" (1.0L)
       : "st(1)");
  acc_split (&log2_x_frac_hi, &log2_x_frac_lo, log2_x_frac_hi, log2_x_frac_lo,
	     log2_1p_w_lo);

  /* Correct for the masking off of X_FRAC_LOW.  */
  acc_split (&log2_x_frac_hi, &log2_x_frac_lo, log2_x_frac_hi, log2_x_frac_lo,
	     x_frac_low * M_LOG2El);

  /* Add the integer and fractional parts of the base-2 logarithm.  */
  long double log2_x_hi, log2_x_lo;
  log2_x_hi = x_int_exponent + log2_x_frac_hi;
  log2_x_lo = ((x_int_exponent - log2_x_hi) + log2_x_frac_hi) + log2_x_frac_lo;

  /* Compute the base-2 logarithm of the result.  */
  long double log2_res_hi, log2_res_lo;
  long double log2_x_hi32, log2_x_lo64;
  GET_LDOUBLE_WORDS (se, i0, i1, log2_x_hi);
  i1 = 0;
  SET_LDOUBLE_WORDS (log2_x_hi32, se, i0, i1);
  log2_x_lo64 = (log2_x_hi - log2_x_hi32) + log2_x_lo;
  long double y_hi32, y_lo32;
  GET_LDOUBLE_WORDS (se, i0, i1, y);
  i1 = 0;
  SET_LDOUBLE_WORDS (y_hi32, se, i0, i1);
  y_lo32 = y - y_hi32;
  log2_res_hi = log2_x_hi32 * y_hi32;
  log2_res_lo = log2_x_hi32 * y_lo32 + log2_x_lo64 * y;

  /* Split the base-2 logarithm of the result into integer and
     fractional parts.  */
  long double log2_res_int = roundl (log2_res_hi);
  long double log2_res_frac = log2_res_hi - log2_res_int + log2_res_lo;
  /* If the integer part is very large, the computed fractional part
     may be outside the valid range for f2xm1.  */
  if (fabsl (log2_res_int) > 16500)
    log2_res_frac = 0;

  /* Compute the final result.  */
  long double res;
  asm ("f2xm1" : "=t" (res) : "0" (log2_res_frac));
  res += 1.0L;
  if (negate)
    res = -res;
  asm ("fscale" : "=t" (res) : "0" (res), "u" (log2_res_int));
  math_check_force_underflow (res);
  return res;
}

libm_hidden_def (__powl_helper)
