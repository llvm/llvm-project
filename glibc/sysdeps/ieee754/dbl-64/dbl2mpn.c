/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>

/* Convert a `double' in IEEE754 standard double-precision format to a
   multi-precision integer representing the significand scaled up by its
   number of bits (52 for double) and an integral power of two (MPN frexp). */

mp_size_t
__mpn_extract_double (mp_ptr res_ptr, mp_size_t size,
		      int *expt, int *is_neg,
		      double value)
{
  union ieee754_double u;
  u.d = value;

  *is_neg = u.ieee.negative;
  *expt = (int) u.ieee.exponent - IEEE754_DOUBLE_BIAS;

#if BITS_PER_MP_LIMB == 32
  res_ptr[0] = u.ieee.mantissa1; /* Low-order 32 bits of fraction.  */
  res_ptr[1] = u.ieee.mantissa0; /* High-order 20 bits.  */
  # define N 2
#elif BITS_PER_MP_LIMB == 64
  /* Hopefully the compiler will combine the two bitfield extracts
     and this composition into just the original quadword extract.  */
  res_ptr[0] = ((mp_limb_t) u.ieee.mantissa0 << 32) | u.ieee.mantissa1;
  # define N 1
#else
  # error "mp_limb size " BITS_PER_MP_LIMB "not accounted for"
#endif
/* The format does not fill the last limb.  There are some zeros.  */
#define NUM_LEADING_ZEROS (BITS_PER_MP_LIMB \
			   - (DBL_MANT_DIG - ((N - 1) * BITS_PER_MP_LIMB)))

  if (u.ieee.exponent == 0)
    {
      /* A biased exponent of zero is a special case.
	 Either it is a zero or it is a denormal number.  */
      if (res_ptr[0] == 0 && res_ptr[N - 1] == 0) /* Assumes N<=2.  */
	/* It's zero.  */
	*expt = 0;
      else
	{
          /* It is a denormal number, meaning it has no implicit leading
	     one bit, and its exponent is in fact the format minimum.  */
	  int cnt;

	  if (res_ptr[N - 1] != 0)
	    {
	      count_leading_zeros (cnt, res_ptr[N - 1]);
	      cnt -= NUM_LEADING_ZEROS;
#if N == 2
	      res_ptr[N - 1] = res_ptr[1] << cnt
			       | (N - 1)
			       * (res_ptr[0] >> (BITS_PER_MP_LIMB - cnt));
	      res_ptr[0] <<= cnt;
#else
	      res_ptr[N - 1] <<= cnt;
#endif
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
	}
    }
  else
    /* Add the implicit leading one bit for a normalized number.  */
    res_ptr[N - 1] |= (mp_limb_t) 1 << (DBL_MANT_DIG - 1
					- ((N - 1) * BITS_PER_MP_LIMB));

  return N;
}
