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

#include <ieee754.h>
#include <errno.h>
#include <float.h>
#include <math.h>

#include "gmp.h"
#include "gmp-impl.h"

/* Convert a multi-precision integer of the needed number of bits (106
   for long double) and an integral power of two to a `long double' in
   IBM extended format.  */

long double
__mpn_construct_long_double (mp_srcptr frac_ptr, int expt, int sign)
{
  union ibm_extended_long_double u;
  unsigned long lzcount;
  unsigned long long hi, lo;
  int exponent2;

  u.d[0].ieee.negative = sign;
  u.d[1].ieee.negative = sign;
  u.d[0].ieee.exponent = expt + IEEE754_DOUBLE_BIAS;
  u.d[1].ieee.exponent = 0;
  exponent2 = expt - 53 + IEEE754_DOUBLE_BIAS;

#if BITS_PER_MP_LIMB == 32
  /* The low order 53 bits (52 + hidden) go into the lower double */
  lo = frac_ptr[0];
  lo |= (frac_ptr[1] & ((1LL << (53 - 32)) - 1)) << 32;
  /* The high order 53 bits (52 + hidden) go into the upper double */
  hi = (frac_ptr[1] >> (53 - 32)) & ((1 << 11) - 1);
  hi |= ((unsigned long long) frac_ptr[2]) << 11;
  hi |= ((unsigned long long) frac_ptr[3]) << (32 + 11);
#elif BITS_PER_MP_LIMB == 64
  /* The low order 53 bits (52 + hidden) go into the lower double */
  lo = frac_ptr[0] & (((mp_limb_t) 1 << 53) - 1);
  /* The high order 53 bits (52 + hidden) go into the upper double */
  hi = (frac_ptr[0] >> 53) & (((mp_limb_t) 1 << 11) - 1);
  hi |= (frac_ptr[1] << 11);
#else
  #error "mp_limb size " BITS_PER_MP_LIMB "not accounted for"
#endif

  if ((hi & (1LL << 52)) == 0 && (hi | lo) != 0)
    {
      /* denormal number  */
      unsigned long long val = hi ? hi : lo;

      if (sizeof (val) == sizeof (long))
	lzcount = __builtin_clzl (val);
      else if ((val >> 32) != 0)
	lzcount = __builtin_clzl ((long) (val >> 32));
      else
	lzcount = __builtin_clzl ((long) val) + 32;
      if (hi)
	lzcount = lzcount - (64 - 53);
      else
	lzcount = lzcount + 53 - (64 - 53);

      if (lzcount > u.d[0].ieee.exponent)
	{
	  lzcount = u.d[0].ieee.exponent;
	  u.d[0].ieee.exponent = 0;
	  exponent2 -= lzcount;
	}
      else
	{
	  u.d[0].ieee.exponent -= (lzcount - 1);
	  exponent2 -= (lzcount - 1);
	}

      if (lzcount <= 53)
	{
	  hi = (hi << lzcount) | (lo >> (53 - lzcount));
	  lo = (lo << lzcount) & ((1LL << 53) - 1);
	}
      else
	{
	  hi = lo << (lzcount - 53);
	  lo = 0;
	}
    }

  if (lo != 0)
    {
      /* hidden bit of low double controls rounding of the high double.
	 If hidden is '1' and either the explicit mantissa is non-zero
	 or hi is odd, then round up hi and adjust lo (2nd mantissa)
	 plus change the sign of the low double to compensate.  */
      if ((lo & (1LL << 52)) != 0
	  && ((hi & 1) != 0 || (lo & ((1LL << 52) - 1)) != 0))
	{
	  hi++;
	  if ((hi & (1LL << 53)) != 0)
	    {
	      hi >>= 1;
	      u.d[0].ieee.exponent++;
	      if (u.d[0].ieee.exponent == IEEE754_DOUBLE_BIAS + DBL_MAX_EXP)
		{
		  /* Overflow.  The appropriate overflowed result must
		     be produced (if an infinity, that means the low
		     part must be zero).  */
		  __set_errno (ERANGE);
		  return (sign ? -LDBL_MAX : LDBL_MAX) * LDBL_MAX;
		}
	    }
	  u.d[1].ieee.negative = !sign;
	  lo = (1LL << 53) - lo;
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
      exponent2 -= lzcount;

      if (exponent2 > 0)
	u.d[1].ieee.exponent = exponent2;
      else if (exponent2 > -53)
	lo >>= 1 - exponent2;
      else
	lo = 0;
    }
  else
    u.d[1].ieee.negative = 0;

  u.d[1].ieee.mantissa1 = lo;
  u.d[1].ieee.mantissa0 = lo >> 32;
  u.d[0].ieee.mantissa1 = hi;
  u.d[0].ieee.mantissa0 = hi >> 32;

  return u.ld;
}
