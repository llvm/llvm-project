/* Manipulation of the bit representation of 'long double' quantities.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_LDBL_H_
#define _MATH_LDBL_H_ 1

#include <ieee754.h>
#include <stdint.h>

/* To suit our callers we return *hi64 and *lo64 as if they came from
   an ieee854 112 bit mantissa, that is, 48 bits in *hi64 (plus one
   implicit bit) and 64 bits in *lo64.  */

static inline void
ldbl_extract_mantissa (int64_t *hi64, uint64_t *lo64, int *exp, long double x)
{
  /* We have 105 bits of mantissa plus one implicit digit.  Since
     106 bits are representable we use the first implicit digit for
     the number before the decimal point and the second implicit bit
     as bit 53 of the mantissa.  */
  uint64_t hi, lo;
  union ibm_extended_long_double u;

  u.ld = x;
  *exp = u.d[0].ieee.exponent - IEEE754_DOUBLE_BIAS;

  lo = ((uint64_t) u.d[1].ieee.mantissa0 << 32) | u.d[1].ieee.mantissa1;
  hi = ((uint64_t) u.d[0].ieee.mantissa0 << 32) | u.d[0].ieee.mantissa1;

  if (u.d[0].ieee.exponent != 0)
    {
      int ediff;

      /* If not a denormal or zero then we have an implicit 53rd bit.  */
      hi |= (uint64_t) 1 << 52;

      if (u.d[1].ieee.exponent != 0)
	lo |= (uint64_t) 1 << 52;
      else
	/* A denormal is to be interpreted as having a biased exponent
	   of 1.  */
	lo = lo << 1;

      /* We are going to shift 4 bits out of hi later, because we only
	 want 48 bits in *hi64.  That means we want 60 bits in lo, but
	 we currently only have 53.  Shift the value up.  */
      lo = lo << 7;

      /* The lower double is normalized separately from the upper.
	 We may need to adjust the lower mantissa to reflect this.
	 The difference between the exponents can be larger than 53
	 when the low double is much less than 1ULP of the upper
	 (in which case there are significant bits, all 0's or all
	 1's, between the two significands).  The difference between
	 the exponents can be less than 53 when the upper double
	 exponent is nearing its minimum value (in which case the low
	 double is denormal ie. has an exponent of zero).  */
      ediff = u.d[0].ieee.exponent - u.d[1].ieee.exponent - 53;
      if (ediff > 0)
	{
	  if (ediff < 64)
	    lo = lo >> ediff;
	  else
	    lo = 0;
	}
      else if (ediff < 0)
	lo = lo << -ediff;

      if (u.d[0].ieee.negative != u.d[1].ieee.negative
	  && lo != 0)
	{
	  hi--;
	  lo = ((uint64_t) 1 << 60) - lo;
	  if (hi < (uint64_t) 1 << 52)
	    {
	      /* We have a borrow from the hidden bit, so shift left 1.  */
	      hi = (hi << 1) | (lo >> 59);
	      lo = (((uint64_t) 1 << 60) - 1) & (lo << 1);
	      *exp = *exp - 1;
	    }
	}
    }
  else
    /* If the larger magnitude double is denormal then the smaller
       one must be zero.  */
    hi = hi << 1;

  *lo64 = (hi << 60) | lo;
  *hi64 = hi >> 4;
}

static inline long double
ldbl_insert_mantissa (int sign, int exp, int64_t hi64, uint64_t lo64)
{
  union ibm_extended_long_double u;
  int expnt2;
  uint64_t hi, lo;

  u.d[0].ieee.negative = sign;
  u.d[1].ieee.negative = sign;
  u.d[0].ieee.exponent = exp + IEEE754_DOUBLE_BIAS;
  u.d[1].ieee.exponent = 0;
  expnt2 = exp - 53 + IEEE754_DOUBLE_BIAS;

  /* Expect 113 bits (112 bits + hidden) right justified in two longs.
     The low order 53 bits (52 + hidden) go into the lower double */
  lo = (lo64 >> 7) & (((uint64_t) 1 << 53) - 1);
  /* The high order 53 bits (52 + hidden) go into the upper double */
  hi = lo64 >> 60;
  hi |= hi64 << 4;

  if (lo != 0)
    {
      int lzcount;

      /* hidden bit of low double controls rounding of the high double.
	 If hidden is '1' and either the explicit mantissa is non-zero
	 or hi is odd, then round up hi and adjust lo (2nd mantissa)
	 plus change the sign of the low double to compensate.  */
      if ((lo & ((uint64_t) 1 << 52)) != 0
	  && ((hi & 1) != 0 || (lo & (((uint64_t) 1 << 52) - 1)) != 0))
	{
	  hi++;
	  if ((hi & ((uint64_t) 1 << 53)) != 0)
	    {
	      hi = hi >> 1;
	      u.d[0].ieee.exponent++;
	    }
	  u.d[1].ieee.negative = !sign;
	  lo = ((uint64_t) 1 << 53) - lo;
	}

      /* Normalize the low double.  Shift the mantissa left until
	 the hidden bit is '1' and adjust the exponent accordingly.  */

      if (sizeof (lo) == sizeof (long))
	lzcount = __builtin_clzl (lo);
      else if ((lo >> 32) != 0)
	lzcount = __builtin_clzl ((long) (lo >> 32));
      else
	lzcount = __builtin_clzl ((long) lo) + 32;
      lzcount = lzcount - (64 - 53);
      lo <<= lzcount;
      expnt2 -= lzcount;

      if (expnt2 >= 1)
	/* Not denormal.  */
	u.d[1].ieee.exponent = expnt2;
      else
	{
	  /* Is denormal.  Note that biased exponent of 0 is treated
	     as if it was 1, hence the extra shift.  */
	  if (expnt2 > -53)
	    lo >>= 1 - expnt2;
	  else
	    lo = 0;
	}
    }
  else
    u.d[1].ieee.negative = 0;

  u.d[1].ieee.mantissa1 = lo;
  u.d[1].ieee.mantissa0 = lo >> 32;
  u.d[0].ieee.mantissa1 = hi;
  u.d[0].ieee.mantissa0 = hi >> 32;
  return u.ld;
}

/* Handy utility functions to pack/unpack/cononicalize and find the nearbyint
   of long double implemented as double double.  */
static inline long double
default_ldbl_pack (double a, double aa)
{
  union ibm_extended_long_double u;
  u.d[0].d = a;
  u.d[1].d = aa;
  return u.ld;
}

static inline void
default_ldbl_unpack (long double l, double *a, double *aa)
{
  union ibm_extended_long_double u;
  u.ld = l;
  *a = u.d[0].d;
  *aa = u.d[1].d;
}

#ifndef ldbl_pack
# define ldbl_pack   default_ldbl_pack
#endif
#ifndef ldbl_unpack
# define ldbl_unpack default_ldbl_unpack
#endif

/* Extract high double.  */
#define ldbl_high(x) ((double) x)

/* Convert a finite long double to canonical form.
   Does not handle +/-Inf properly.  */
static inline void
ldbl_canonicalize (double *a, double *aa)
{
  double xh, xl;

  xh = *a + *aa;
  xl = (*a - xh) + *aa;
  *a = xh;
  *aa = xl;
}

/* Simple inline nearbyint (double) function.
   Only works in the default rounding mode
   but is useful in long double rounding functions.  */
static inline double
ldbl_nearbyint (double a)
{
  double two52 = 0x1p52;

  if (__glibc_likely ((__builtin_fabs (a) < two52)))
    {
      if (__glibc_likely ((a > 0.0)))
	{
	  a += two52;
	  a -= two52;
	}
      else if (__glibc_likely ((a < 0.0)))
	{
	  a = two52 - a;
	  a = -(a - two52);
	}
    }
  return a;
}

/* Canonicalize a result from an integer rounding function, in any
   rounding mode.  *A and *AA are finite and integers, with *A being
   nonzero; if the result is not already canonical, *AA is plus or
   minus a power of 2 that does not exceed the least set bit in
   *A.  */
static inline void
ldbl_canonicalize_int (double *a, double *aa)
{
  /* Previously we used EXTRACT_WORDS64 from math_private.h, but in order
     to avoid including internal headers we duplicate that code here.  */
  uint64_t ax, aax;
  union { double value; uint64_t word; } extractor;
  extractor.value = *a;
  ax = extractor.word;
  extractor.value = *aa;
  aax = extractor.word;

  int expdiff = ((ax >> 52) & 0x7ff) - ((aax >> 52) & 0x7ff);
  if (expdiff <= 53)
    {
      if (expdiff == 53)
	{
	  /* Half way between two double values; noncanonical iff the
	     low bit of A's mantissa is 1.  */
	  if ((ax & 1) != 0)
	    {
	      *a += 2 * *aa;
	      *aa = -*aa;
	    }
	}
      else
	{
	  /* The sum can be represented in a single double.  */
	  *a += *aa;
	  *aa = 0;
	}
    }
}

#endif /* math_ldbl.h */
