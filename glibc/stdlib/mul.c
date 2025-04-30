/* mpn_mul -- Multiply two natural numbers.

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

/* Multiply the natural numbers u (pointed to by UP, with USIZE limbs)
   and v (pointed to by VP, with VSIZE limbs), and store the result at
   PRODP.  USIZE + VSIZE limbs are always stored, but if the input
   operands are normalized.  Return the most significant limb of the
   result.

   NOTE: The space pointed to by PRODP is overwritten before finished
   with U and V, so overlap is an error.

   Argument constraints:
   1. USIZE >= VSIZE.
   2. PRODP != UP and PRODP != VP, i.e. the destination
      must be distinct from the multiplier and the multiplicand.  */

/* If KARATSUBA_THRESHOLD is not already defined, define it to a
   value which is good on most machines.  */
#ifndef KARATSUBA_THRESHOLD
#define KARATSUBA_THRESHOLD 32
#endif

mp_limb_t
mpn_mul (mp_ptr prodp,
	 mp_srcptr up, mp_size_t usize,
	 mp_srcptr vp, mp_size_t vsize)
{
  mp_ptr prod_endp = prodp + usize + vsize - 1;
  mp_limb_t cy;
  mp_ptr tspace;
  TMP_DECL (marker);

  if (vsize < KARATSUBA_THRESHOLD)
    {
      /* Handle simple cases with traditional multiplication.

	 This is the most critical code of the entire function.  All
	 multiplies rely on this, both small and huge.  Small ones arrive
	 here immediately.  Huge ones arrive here as this is the base case
	 for Karatsuba's recursive algorithm below.  */
      mp_size_t i;
      mp_limb_t cy_limb;
      mp_limb_t v_limb;

      if (vsize == 0)
	return 0;

      /* Multiply by the first limb in V separately, as the result can be
	 stored (not added) to PROD.  We also avoid a loop for zeroing.  */
      v_limb = vp[0];
      if (v_limb <= 1)
	{
	  if (v_limb == 1)
	    MPN_COPY (prodp, up, usize);
	  else
	    MPN_ZERO (prodp, usize);
	  cy_limb = 0;
	}
      else
	cy_limb = mpn_mul_1 (prodp, up, usize, v_limb);

      prodp[usize] = cy_limb;
      prodp++;

      /* For each iteration in the outer loop, multiply one limb from
	 U with one limb from V, and add it to PROD.  */
      for (i = 1; i < vsize; i++)
	{
	  v_limb = vp[i];
	  if (v_limb <= 1)
	    {
	      cy_limb = 0;
	      if (v_limb == 1)
		cy_limb = mpn_add_n (prodp, prodp, up, usize);
	    }
	  else
	    cy_limb = mpn_addmul_1 (prodp, up, usize, v_limb);

	  prodp[usize] = cy_limb;
	  prodp++;
	}
      return cy_limb;
    }

  TMP_MARK (marker);

  tspace = (mp_ptr) TMP_ALLOC (2 * vsize * BYTES_PER_MP_LIMB);
  MPN_MUL_N_RECURSE (prodp, up, vp, vsize, tspace);

  prodp += vsize;
  up += vsize;
  usize -= vsize;
  if (usize >= vsize)
    {
      mp_ptr tp = (mp_ptr) TMP_ALLOC (2 * vsize * BYTES_PER_MP_LIMB);
      do
	{
	  MPN_MUL_N_RECURSE (tp, up, vp, vsize, tspace);
	  cy = mpn_add_n (prodp, prodp, tp, vsize);
	  mpn_add_1 (prodp + vsize, tp + vsize, vsize, cy);
	  prodp += vsize;
	  up += vsize;
	  usize -= vsize;
	}
      while (usize >= vsize);
    }

  /* True: usize < vsize.  */

  /* Make life simple: Recurse.  */

  if (usize != 0)
    {
      mpn_mul (tspace, vp, vsize, up, usize);
      cy = mpn_add_n (prodp, prodp, tspace, vsize);
      mpn_add_1 (prodp + vsize, tspace + vsize, usize, cy);
    }

  TMP_FREE (marker);
  return *prod_endp;
}
