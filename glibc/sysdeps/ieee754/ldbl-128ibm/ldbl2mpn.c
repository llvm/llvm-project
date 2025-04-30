/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"
#include <ieee754.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

/* Convert a `long double' in IBM extended format to a multi-precision
   integer representing the significand scaled up by its number of
   bits (106 for long double) and an integral power of two (MPN
   frexpl). */


/* When signs differ, the actual value is the difference between the
   significant double and the less significant double.  Sometimes a
   bit can be lost when we borrow from the significant mantissa.  */
#define EXTRA_INTERNAL_PRECISION (7)

mp_size_t
__mpn_extract_long_double (mp_ptr res_ptr, mp_size_t size,
			   int *expt, int *is_neg,
			   long double value)
{
  union ibm_extended_long_double u;
  unsigned long long hi, lo;
  int ediff;

  u.ld = value;

  *is_neg = u.d[0].ieee.negative;
  *expt = (int) u.d[0].ieee.exponent - IEEE754_DOUBLE_BIAS;

  lo = ((long long) u.d[1].ieee.mantissa0 << 32) | u.d[1].ieee.mantissa1;
  hi = ((long long) u.d[0].ieee.mantissa0 << 32) | u.d[0].ieee.mantissa1;

  /* Hold 7 extra bits of precision in the mantissa.  This allows
     the normalizing shifts below to prevent losing precision when
     the signs differ and the exponents are sufficiently far apart.  */
  lo <<= EXTRA_INTERNAL_PRECISION;

  /* If the lower double is not a denormal or zero then set the hidden
     53rd bit.  */
  if (u.d[1].ieee.exponent != 0)
    lo |= 1ULL << (52 + EXTRA_INTERNAL_PRECISION);
  else
    lo = lo << 1;

  /* The lower double is normalized separately from the upper.  We may
     need to adjust the lower manitissa to reflect this.  */
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

  /* The high double may be rounded and the low double reflects the
     difference between the long double and the rounded high double
     value.  This is indicated by a differnce between the signs of the
     high and low doubles.  */
  if (u.d[0].ieee.negative != u.d[1].ieee.negative
      && lo != 0)
    {
      lo = (1ULL << (53 + EXTRA_INTERNAL_PRECISION)) - lo;
      if (hi == 0)
	{
	  /* we have a borrow from the hidden bit, so shift left 1.  */
	  hi = 0x000ffffffffffffeLL | (lo >> (52 + EXTRA_INTERNAL_PRECISION));
	  lo = 0x0fffffffffffffffLL & (lo << 1);
	  (*expt)--;
	}
      else
	hi--;
    }
#if BITS_PER_MP_LIMB == 32
  /* Combine the mantissas to be contiguous.  */
  res_ptr[0] = lo >> EXTRA_INTERNAL_PRECISION;
  res_ptr[1] = (hi << (53 - 32)) | (lo >> (32 + EXTRA_INTERNAL_PRECISION));
  res_ptr[2] = hi >> 11;
  res_ptr[3] = hi >> (32 + 11);
  #define N 4
#elif BITS_PER_MP_LIMB == 64
  /* Combine the two mantissas to be contiguous.  */
  res_ptr[0] = (hi << 53) | (lo >> EXTRA_INTERNAL_PRECISION);
  res_ptr[1] = hi >> 11;
  #define N 2
#else
  #error "mp_limb size " BITS_PER_MP_LIMB "not accounted for"
#endif
/* The format does not fill the last limb.  There are some zeros.  */
#define NUM_LEADING_ZEROS (BITS_PER_MP_LIMB \
			   - (LDBL_MANT_DIG - ((N - 1) * BITS_PER_MP_LIMB)))

  if (u.d[0].ieee.exponent == 0)
    {
      /* A biased exponent of zero is a special case.
	 Either it is a zero or it is a denormal number.  */
      if (res_ptr[0] == 0 && res_ptr[1] == 0
	  && res_ptr[N - 2] == 0 && res_ptr[N - 1] == 0) /* Assumes N<=4.  */
	/* It's zero.  */
	*expt = 0;
      else
	{
	  /* It is a denormal number, meaning it has no implicit leading
	     one bit, and its exponent is in fact the format minimum.  We
	     use DBL_MIN_EXP instead of LDBL_MIN_EXP below because the
	     latter describes the properties of both parts together, but
	     the exponent is computed from the high part only.  */
	  int cnt;

#if N == 2
	  if (res_ptr[N - 1] != 0)
	    {
	      count_leading_zeros (cnt, res_ptr[N - 1]);
	      cnt -= NUM_LEADING_ZEROS;
	      res_ptr[N - 1] = res_ptr[N - 1] << cnt
			       | (res_ptr[0] >> (BITS_PER_MP_LIMB - cnt));
	      res_ptr[0] <<= cnt;
	      *expt = DBL_MIN_EXP - 1 - cnt;
	    }
	  else
	    {
	      count_leading_zeros (cnt, res_ptr[0]);
	      if (cnt >= NUM_LEADING_ZEROS)
		{
		  res_ptr[N - 1] = res_ptr[0] << (cnt - NUM_LEADING_ZEROS);
		  res_ptr[0] = 0;
		}
	      else
		{
		  res_ptr[N - 1] = res_ptr[0] >> (NUM_LEADING_ZEROS - cnt);
		  res_ptr[0] <<= BITS_PER_MP_LIMB - (NUM_LEADING_ZEROS - cnt);
		}
	      *expt = DBL_MIN_EXP - 1
		- (BITS_PER_MP_LIMB - NUM_LEADING_ZEROS) - cnt;
	    }
#else
	  int j, k, l;

	  for (j = N - 1; j > 0; j--)
	    if (res_ptr[j] != 0)
	      break;

	  count_leading_zeros (cnt, res_ptr[j]);
	  cnt -= NUM_LEADING_ZEROS;
	  l = N - 1 - j;
	  if (cnt < 0)
	    {
	      cnt += BITS_PER_MP_LIMB;
	      l--;
	    }
	  if (!cnt)
	    for (k = N - 1; k >= l; k--)
	      res_ptr[k] = res_ptr[k-l];
	  else
	    {
	      for (k = N - 1; k > l; k--)
		res_ptr[k] = res_ptr[k-l] << cnt
			     | res_ptr[k-l-1] >> (BITS_PER_MP_LIMB - cnt);
	      res_ptr[k--] = res_ptr[0] << cnt;
	    }

	  for (; k >= 0; k--)
	    res_ptr[k] = 0;
	  *expt = DBL_MIN_EXP - 1 - l * BITS_PER_MP_LIMB - cnt;
#endif
	}
    }
  else
    /* Add the implicit leading one bit for a normalized number.  */
    res_ptr[N - 1] |= (mp_limb_t) 1 << (LDBL_MANT_DIG - 1
					- ((N - 1) * BITS_PER_MP_LIMB));

  return N;
}
