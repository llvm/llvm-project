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
#include <stdlib.h>

/* Convert a `long double' in IEEE854 standard double-precision format to a
   multi-precision integer representing the significand scaled up by its
   number of bits (64 for long double) and an integral power of two
   (MPN frexpl). */

mp_size_t
__mpn_extract_long_double (mp_ptr res_ptr, mp_size_t size,
			   int *expt, int *is_neg,
			   long double value)
{
  union ieee854_long_double u;
  u.d = value;

  *is_neg = u.ieee.negative;
  *expt = (int) u.ieee.exponent - IEEE854_LONG_DOUBLE_BIAS;

#if BITS_PER_MP_LIMB == 32
  res_ptr[0] = u.ieee.mantissa1; /* Low-order 32 bits of fraction.  */
  res_ptr[1] = u.ieee.mantissa0; /* High-order 32 bits.  */
  #define N 2
#elif BITS_PER_MP_LIMB == 64
  /* Hopefully the compiler will combine the two bitfield extracts
     and this composition into just the original quadword extract.  */
  res_ptr[0] = ((mp_limb_t) u.ieee.mantissa0 << 32) | u.ieee.mantissa1;
  #define N 1
#else
  #error "mp_limb size " BITS_PER_MP_LIMB "not accounted for"
#endif

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

	  /* One problem with Intel's 80-bit format is that the explicit
	     leading one in the normalized representation has to be zero
	     for denormalized number.  If it is one, the number is according
	     to Intel's specification an invalid number.  We make the
	     representation unique by explicitly clearing this bit.  */
	  res_ptr[N - 1] &= ~((mp_limb_t) 1 << ((LDBL_MANT_DIG - 1) % BITS_PER_MP_LIMB));

	  if (res_ptr[N - 1] != 0)
	    {
	      count_leading_zeros (cnt, res_ptr[N - 1]);
	      if (cnt != 0)
		{
#if N == 2
		  res_ptr[N - 1] = res_ptr[N - 1] << cnt
				   | (res_ptr[0] >> (BITS_PER_MP_LIMB - cnt));
		  res_ptr[0] <<= cnt;
#else
		  res_ptr[N - 1] <<= cnt;
#endif
		}
	      *expt = LDBL_MIN_EXP - 1 - cnt;
	    }
	  else if (res_ptr[0] != 0)
	    {
	      count_leading_zeros (cnt, res_ptr[0]);
	      res_ptr[N - 1] = res_ptr[0] << cnt;
	      res_ptr[0] = 0;
	      *expt = LDBL_MIN_EXP - 1 - BITS_PER_MP_LIMB - cnt;
	    }
	  else
	    {
	      /* This is the special case of the pseudo denormal number
		 with only the implicit leading bit set.  The value is
		 in fact a normal number and so we have to treat this
		 case differently.  */
#if N == 2
	      res_ptr[N - 1] = 0x80000000ul;
#else
	      res_ptr[0] = 0x8000000000000000ul;
#endif
	      *expt = LDBL_MIN_EXP - 1;
	    }
	}
    }
  else if (u.ieee.exponent < 0x7fff
#if N == 2
	   && res_ptr[0] == 0
#endif
	   && res_ptr[N - 1] == 0)
    /* Pseudo zero.  */
    *expt = 0;

  return N;
}
