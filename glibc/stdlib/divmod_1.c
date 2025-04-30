/* mpn_divmod_1(quot_ptr, dividend_ptr, dividend_size, divisor_limb) --
   Divide (DIVIDEND_PTR,,DIVIDEND_SIZE) by DIVISOR_LIMB.
   Write DIVIDEND_SIZE limbs of quotient at QUOT_PTR.
   Return the single-limb remainder.
   There are no constraints on the value of the divisor.

   QUOT_PTR and DIVIDEND_PTR might point to the same limb.

Copyright (C) 1991-2021 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MP Library; see the file COPYING.LIB.  If not, see
<https://www.gnu.org/licenses/>.  */

#include <gmp.h>
#include "gmp-impl.h"
#include "longlong.h"

#ifndef UMUL_TIME
#define UMUL_TIME 1
#endif

#ifndef UDIV_TIME
#define UDIV_TIME UMUL_TIME
#endif

/* FIXME: We should be using invert_limb (or invert_normalized_limb)
   here (not udiv_qrnnd).  */

mp_limb_t
mpn_divmod_1 (mp_ptr quot_ptr,
	      mp_srcptr dividend_ptr, mp_size_t dividend_size,
	      mp_limb_t divisor_limb)
{
  mp_size_t i;
  mp_limb_t n1, n0, r;
  mp_limb_t dummy __attribute__ ((unused));

  /* ??? Should this be handled at all?  Rely on callers?  */
  if (dividend_size == 0)
    return 0;

  /* If multiplication is much faster than division, and the
     dividend is large, pre-invert the divisor, and use
     only multiplications in the inner loop.  */

  /* This test should be read:
       Does it ever help to use udiv_qrnnd_preinv?
	 && Does what we save compensate for the inversion overhead?  */
  if (UDIV_TIME > (2 * UMUL_TIME + 6)
      && (UDIV_TIME - (2 * UMUL_TIME + 6)) * dividend_size > UDIV_TIME)
    {
      int normalization_steps;

      count_leading_zeros (normalization_steps, divisor_limb);
      if (normalization_steps != 0)
	{
	  mp_limb_t divisor_limb_inverted;

	  divisor_limb <<= normalization_steps;

	  /* Compute (2**2N - 2**N * DIVISOR_LIMB) / DIVISOR_LIMB.  The
	     result is a (N+1)-bit approximation to 1/DIVISOR_LIMB, with the
	     most significant bit (with weight 2**N) implicit.  */

	  /* Special case for DIVISOR_LIMB == 100...000.  */
	  if (divisor_limb << 1 == 0)
	    divisor_limb_inverted = ~(mp_limb_t) 0;
	  else
	    udiv_qrnnd (divisor_limb_inverted, dummy,
			-divisor_limb, 0, divisor_limb);

	  n1 = dividend_ptr[dividend_size - 1];
	  r = n1 >> (BITS_PER_MP_LIMB - normalization_steps);

	  /* Possible optimization:
	     if (r == 0
	     && divisor_limb > ((n1 << normalization_steps)
			     | (dividend_ptr[dividend_size - 2] >> ...)))
	     ...one division less... */

	  for (i = dividend_size - 2; i >= 0; i--)
	    {
	      n0 = dividend_ptr[i];
	      udiv_qrnnd_preinv (quot_ptr[i + 1], r, r,
				 ((n1 << normalization_steps)
				  | (n0 >> (BITS_PER_MP_LIMB - normalization_steps))),
				 divisor_limb, divisor_limb_inverted);
	      n1 = n0;
	    }
	  udiv_qrnnd_preinv (quot_ptr[0], r, r,
			     n1 << normalization_steps,
			     divisor_limb, divisor_limb_inverted);
	  return r >> normalization_steps;
	}
      else
	{
	  mp_limb_t divisor_limb_inverted;

	  /* Compute (2**2N - 2**N * DIVISOR_LIMB) / DIVISOR_LIMB.  The
	     result is a (N+1)-bit approximation to 1/DIVISOR_LIMB, with the
	     most significant bit (with weight 2**N) implicit.  */

	  /* Special case for DIVISOR_LIMB == 100...000.  */
	  if (divisor_limb << 1 == 0)
	    divisor_limb_inverted = ~(mp_limb_t) 0;
	  else
	    udiv_qrnnd (divisor_limb_inverted, dummy,
			-divisor_limb, 0, divisor_limb);

	  i = dividend_size - 1;
	  r = dividend_ptr[i];

	  if (r >= divisor_limb)
	    r = 0;
	  else
	    {
	      quot_ptr[i] = 0;
	      i--;
	    }

	  for (; i >= 0; i--)
	    {
	      n0 = dividend_ptr[i];
	      udiv_qrnnd_preinv (quot_ptr[i], r, r,
				 n0, divisor_limb, divisor_limb_inverted);
	    }
	  return r;
	}
    }
  else
    {
      if (UDIV_NEEDS_NORMALIZATION)
	{
	  int normalization_steps;

	  count_leading_zeros (normalization_steps, divisor_limb);
	  if (normalization_steps != 0)
	    {
	      divisor_limb <<= normalization_steps;

	      n1 = dividend_ptr[dividend_size - 1];
	      r = n1 >> (BITS_PER_MP_LIMB - normalization_steps);

	      /* Possible optimization:
		 if (r == 0
		 && divisor_limb > ((n1 << normalization_steps)
				 | (dividend_ptr[dividend_size - 2] >> ...)))
		 ...one division less... */

	      for (i = dividend_size - 2; i >= 0; i--)
		{
		  n0 = dividend_ptr[i];
		  udiv_qrnnd (quot_ptr[i + 1], r, r,
			      ((n1 << normalization_steps)
			       | (n0 >> (BITS_PER_MP_LIMB - normalization_steps))),
			      divisor_limb);
		  n1 = n0;
		}
	      udiv_qrnnd (quot_ptr[0], r, r,
			  n1 << normalization_steps,
			  divisor_limb);
	      return r >> normalization_steps;
	    }
	}
      /* No normalization needed, either because udiv_qrnnd doesn't require
	 it, or because DIVISOR_LIMB is already normalized.  */

      i = dividend_size - 1;
      r = dividend_ptr[i];

      if (r >= divisor_limb)
	r = 0;
      else
	{
	  quot_ptr[i] = 0;
	  i--;
	}

      for (; i >= 0; i--)
	{
	  n0 = dividend_ptr[i];
	  udiv_qrnnd (quot_ptr[i], r, r, n0, divisor_limb);
	}
      return r;
    }
}
