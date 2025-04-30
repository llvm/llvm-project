/* Compute full X * Y for double type.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#ifndef _MUL_SPLIT_H
#define _MUL_SPLIT_H

#include <float.h>

/* Calculate X * Y exactly and store the result in *HI + *LO.  It is
   given that the values are small enough that no overflow occurs and
   large enough (or zero) that no underflow occurs.  */

static void
mul_split (double *hi, double *lo, double x, double y)
{
#ifdef __FP_FAST_FMA
  /* Fast built-in fused multiply-add.  */
  *hi = x * y;
  *lo = __builtin_fma (x, y, -*hi);
#else
  /* Apply Dekker's algorithm.  */
  *hi = x * y;
# define C ((1 << (DBL_MANT_DIG + 1) / 2) + 1)
  double x1 = x * C;
  double y1 = y * C;
# undef C
  x1 = (x - x1) + x1;
  y1 = (y - y1) + y1;
  double x2 = x - x1;
  double y2 = y - y1;
  *lo = (((x1 * y1 - *hi) + x1 * y2) + x2 * y1) + x2 * y2;
#endif
}

/* Add a + b exactly, such that *hi + *lo = a + b.
   Assumes |a| >= |b| and rounding to nearest.  */
static inline void
fast_two_sum (double *hi, double *lo, double a, double b)
{
  double e;

  *hi = a + b;
  e = *hi - a; /* exact  */
  *lo = b - e; /* exact  */
  /* Now *hi + *lo = a + b exactly.  */
}

/* Multiplication of two floating-point expansions: *hi + *lo is an
   approximation of (h1+l1)*(h2+l2), assuming |l1| <= 1/2*ulp(h1)
   and |l2| <= 1/2*ulp(h2) and rounding to nearest.  */
static inline void
mul_expansion (double *hi, double *lo, double h1, double l1,
	       double h2, double l2)
{
  double r, e;

  mul_split (hi, lo, h1, h2);
  r = h1 * l2 + h2 * l1;
  /* Now add r to (hi,lo) using fast two-sum, where we know |r| < |hi|.  */
  fast_two_sum (hi, &e, *hi, r);
  *lo -= e;
}

/* Calculate X / Y and store the approximate result in *HI + *LO.  It is
   assumed that Y is not zero, that no overflow nor underflow occurs, and
   rounding is to nearest.  */
static inline void
div_split (double *hi, double *lo, double x, double y)
{
  double a, b;

  *hi = x / y;
  mul_split (&a, &b, *hi, y);
  /* a + b = hi*y, which should be near x.  */
  a = x - a; /* huge cancellation  */
  a = a - b;
  /* Now x ~ hi*y + a thus x/y ~ hi + a/y.  */
  *lo = a / y;
}

/* Division of two floating-point expansions: *hi + *lo is an
   approximation of (h1+l1)/(h2+l2), assuming |l1| <= 1/2*ulp(h1)
   and |l2| <= 1/2*ulp(h2), h2+l2 is not zero, and rounding to nearest.  */
static inline void
div_expansion (double *hi, double *lo, double h1, double l1,
	       double h2, double l2)
{
  double r, e;

  div_split (hi, lo, h1, h2);
  /* (h1+l1)/(h2+l2) ~ h1/h2 + (l1*h2 - l2*h1)/h2^2  */
  r = (l1 * h2 - l2 * h1) / (h2 * h2);
  /* Now add r to (hi,lo) using fast two-sum, where we know |r| < |hi|.  */
  fast_two_sum (hi, &e, *hi, r);
  *lo += e;
  /* Renormalize since |lo| might be larger than 0.5 ulp(hi).  */
  fast_two_sum (hi, lo, *hi, *lo);
}

#endif /* _MUL_SPLIT_H */
