/* mpn_mul_n -- Multiply two natural numbers of length n.

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

/* Multiply the natural numbers u (pointed to by UP) and v (pointed to by VP),
   both with SIZE limbs, and store the result at PRODP.  2 * SIZE limbs are
   always stored.  Return the most significant limb.

   Argument constraints:
   1. PRODP != UP and PRODP != VP, i.e. the destination
      must be distinct from the multiplier and the multiplicand.  */

/* If KARATSUBA_THRESHOLD is not already defined, define it to a
   value which is good on most machines.  */
#ifndef KARATSUBA_THRESHOLD
#define KARATSUBA_THRESHOLD 32
#endif

/* The code can't handle KARATSUBA_THRESHOLD smaller than 2.  */
#if KARATSUBA_THRESHOLD < 2
#undef KARATSUBA_THRESHOLD
#define KARATSUBA_THRESHOLD 2
#endif

/* Handle simple cases with traditional multiplication.

   This is the most critical code of multiplication.  All multiplies rely
   on this, both small and huge.  Small ones arrive here immediately.  Huge
   ones arrive here as this is the base case for Karatsuba's recursive
   algorithm below.  */

void
impn_mul_n_basecase (mp_ptr prodp, mp_srcptr up, mp_srcptr vp, mp_size_t size)
{
  mp_size_t i;
  mp_limb_t cy_limb;
  mp_limb_t v_limb;

  /* Multiply by the first limb in V separately, as the result can be
     stored (not added) to PROD.  We also avoid a loop for zeroing.  */
  v_limb = vp[0];
  if (v_limb <= 1)
    {
      if (v_limb == 1)
	MPN_COPY (prodp, up, size);
      else
	MPN_ZERO (prodp, size);
      cy_limb = 0;
    }
  else
    cy_limb = mpn_mul_1 (prodp, up, size, v_limb);

  prodp[size] = cy_limb;
  prodp++;

  /* For each iteration in the outer loop, multiply one limb from
     U with one limb from V, and add it to PROD.  */
  for (i = 1; i < size; i++)
    {
      v_limb = vp[i];
      if (v_limb <= 1)
	{
	  cy_limb = 0;
	  if (v_limb == 1)
	    cy_limb = mpn_add_n (prodp, prodp, up, size);
	}
      else
	cy_limb = mpn_addmul_1 (prodp, up, size, v_limb);

      prodp[size] = cy_limb;
      prodp++;
    }
}

void
impn_mul_n (mp_ptr prodp,
	     mp_srcptr up, mp_srcptr vp, mp_size_t size, mp_ptr tspace)
{
  if ((size & 1) != 0)
    {
      /* The size is odd, the code code below doesn't handle that.
	 Multiply the least significant (size - 1) limbs with a recursive
	 call, and handle the most significant limb of S1 and S2
	 separately.  */
      /* A slightly faster way to do this would be to make the Karatsuba
	 code below behave as if the size were even, and let it check for
	 odd size in the end.  I.e., in essence move this code to the end.
	 Doing so would save us a recursive call, and potentially make the
	 stack grow a lot less.  */

      mp_size_t esize = size - 1;	/* even size */
      mp_limb_t cy_limb;

      MPN_MUL_N_RECURSE (prodp, up, vp, esize, tspace);
      cy_limb = mpn_addmul_1 (prodp + esize, up, esize, vp[esize]);
      prodp[esize + esize] = cy_limb;
      cy_limb = mpn_addmul_1 (prodp + esize, vp, size, up[esize]);

      prodp[esize + size] = cy_limb;
    }
  else
    {
      /* Anatolij Alekseevich Karatsuba's divide-and-conquer algorithm.

	 Split U in two pieces, U1 and U0, such that
	 U = U0 + U1*(B**n),
	 and V in V1 and V0, such that
	 V = V0 + V1*(B**n).

	 UV is then computed recursively using the identity

		2n   n          n                     n
	 UV = (B  + B )U V  +  B (U -U )(V -V )  +  (B + 1)U V
			1 1        1  0   0  1              0 0

	 Where B = 2**BITS_PER_MP_LIMB.  */

      mp_size_t hsize = size >> 1;
      mp_limb_t cy;
      int negflg;

      /*** Product H.	 ________________  ________________
			|_____U1 x V1____||____U0 x V0_____|  */
      /* Put result in upper part of PROD and pass low part of TSPACE
	 as new TSPACE.  */
      MPN_MUL_N_RECURSE (prodp + size, up + hsize, vp + hsize, hsize, tspace);

      /*** Product M.	 ________________
			|_(U1-U0)(V0-V1)_|  */
      if (mpn_cmp (up + hsize, up, hsize) >= 0)
	{
	  mpn_sub_n (prodp, up + hsize, up, hsize);
	  negflg = 0;
	}
      else
	{
	  mpn_sub_n (prodp, up, up + hsize, hsize);
	  negflg = 1;
	}
      if (mpn_cmp (vp + hsize, vp, hsize) >= 0)
	{
	  mpn_sub_n (prodp + hsize, vp + hsize, vp, hsize);
	  negflg ^= 1;
	}
      else
	{
	  mpn_sub_n (prodp + hsize, vp, vp + hsize, hsize);
	  /* No change of NEGFLG.  */
	}
      /* Read temporary operands from low part of PROD.
	 Put result in low part of TSPACE using upper part of TSPACE
	 as new TSPACE.  */
      MPN_MUL_N_RECURSE (tspace, prodp, prodp + hsize, hsize, tspace + size);

      /*** Add/copy product H.  */
      MPN_COPY (prodp + hsize, prodp + size, hsize);
      cy = mpn_add_n (prodp + size, prodp + size, prodp + size + hsize, hsize);

      /*** Add product M (if NEGFLG M is a negative number).  */
      if (negflg)
	cy -= mpn_sub_n (prodp + hsize, prodp + hsize, tspace, size);
      else
	cy += mpn_add_n (prodp + hsize, prodp + hsize, tspace, size);

      /*** Product L.	 ________________  ________________
			|________________||____U0 x V0_____|  */
      /* Read temporary operands from low part of PROD.
	 Put result in low part of TSPACE using upper part of TSPACE
	 as new TSPACE.  */
      MPN_MUL_N_RECURSE (tspace, up, vp, hsize, tspace + size);

      /*** Add/copy Product L (twice).  */

      cy += mpn_add_n (prodp + hsize, prodp + hsize, tspace, size);
      if (cy)
	mpn_add_1 (prodp + hsize + size, prodp + hsize + size, hsize, cy);

      MPN_COPY (prodp, tspace, hsize);
      cy = mpn_add_n (prodp + hsize, prodp + hsize, tspace + hsize, hsize);
      if (cy)
	mpn_add_1 (prodp + size, prodp + size, size, 1);
    }
}

void
impn_sqr_n_basecase (mp_ptr prodp, mp_srcptr up, mp_size_t size)
{
  mp_size_t i;
  mp_limb_t cy_limb;
  mp_limb_t v_limb;

  /* Multiply by the first limb in V separately, as the result can be
     stored (not added) to PROD.  We also avoid a loop for zeroing.  */
  v_limb = up[0];
  if (v_limb <= 1)
    {
      if (v_limb == 1)
	MPN_COPY (prodp, up, size);
      else
	MPN_ZERO (prodp, size);
      cy_limb = 0;
    }
  else
    cy_limb = mpn_mul_1 (prodp, up, size, v_limb);

  prodp[size] = cy_limb;
  prodp++;

  /* For each iteration in the outer loop, multiply one limb from
     U with one limb from V, and add it to PROD.  */
  for (i = 1; i < size; i++)
    {
      v_limb = up[i];
      if (v_limb <= 1)
	{
	  cy_limb = 0;
	  if (v_limb == 1)
	    cy_limb = mpn_add_n (prodp, prodp, up, size);
	}
      else
	cy_limb = mpn_addmul_1 (prodp, up, size, v_limb);

      prodp[size] = cy_limb;
      prodp++;
    }
}

void
impn_sqr_n (mp_ptr prodp,
	     mp_srcptr up, mp_size_t size, mp_ptr tspace)
{
  if ((size & 1) != 0)
    {
      /* The size is odd, the code code below doesn't handle that.
	 Multiply the least significant (size - 1) limbs with a recursive
	 call, and handle the most significant limb of S1 and S2
	 separately.  */
      /* A slightly faster way to do this would be to make the Karatsuba
	 code below behave as if the size were even, and let it check for
	 odd size in the end.  I.e., in essence move this code to the end.
	 Doing so would save us a recursive call, and potentially make the
	 stack grow a lot less.  */

      mp_size_t esize = size - 1;	/* even size */
      mp_limb_t cy_limb;

      MPN_SQR_N_RECURSE (prodp, up, esize, tspace);
      cy_limb = mpn_addmul_1 (prodp + esize, up, esize, up[esize]);
      prodp[esize + esize] = cy_limb;
      cy_limb = mpn_addmul_1 (prodp + esize, up, size, up[esize]);

      prodp[esize + size] = cy_limb;
    }
  else
    {
      mp_size_t hsize = size >> 1;
      mp_limb_t cy;

      /*** Product H.	 ________________  ________________
			|_____U1 x U1____||____U0 x U0_____|  */
      /* Put result in upper part of PROD and pass low part of TSPACE
	 as new TSPACE.  */
      MPN_SQR_N_RECURSE (prodp + size, up + hsize, hsize, tspace);

      /*** Product M.	 ________________
			|_(U1-U0)(U0-U1)_|  */
      if (mpn_cmp (up + hsize, up, hsize) >= 0)
	{
	  mpn_sub_n (prodp, up + hsize, up, hsize);
	}
      else
	{
	  mpn_sub_n (prodp, up, up + hsize, hsize);
	}

      /* Read temporary operands from low part of PROD.
	 Put result in low part of TSPACE using upper part of TSPACE
	 as new TSPACE.  */
      MPN_SQR_N_RECURSE (tspace, prodp, hsize, tspace + size);

      /*** Add/copy product H.  */
      MPN_COPY (prodp + hsize, prodp + size, hsize);
      cy = mpn_add_n (prodp + size, prodp + size, prodp + size + hsize, hsize);

      /*** Add product M (if NEGFLG M is a negative number).  */
      cy -= mpn_sub_n (prodp + hsize, prodp + hsize, tspace, size);

      /*** Product L.	 ________________  ________________
			|________________||____U0 x U0_____|  */
      /* Read temporary operands from low part of PROD.
	 Put result in low part of TSPACE using upper part of TSPACE
	 as new TSPACE.  */
      MPN_SQR_N_RECURSE (tspace, up, hsize, tspace + size);

      /*** Add/copy Product L (twice).  */

      cy += mpn_add_n (prodp + hsize, prodp + hsize, tspace, size);
      if (cy)
	mpn_add_1 (prodp + hsize + size, prodp + hsize + size, hsize, cy);

      MPN_COPY (prodp, tspace, hsize);
      cy = mpn_add_n (prodp + hsize, prodp + hsize, tspace + hsize, hsize);
      if (cy)
	mpn_add_1 (prodp + size, prodp + size, size, 1);
    }
}

/* This should be made into an inline function in gmp.h.  */
void
mpn_mul_n (mp_ptr prodp, mp_srcptr up, mp_srcptr vp, mp_size_t size)
{
  TMP_DECL (marker);
  TMP_MARK (marker);
  if (up == vp)
    {
      if (size < KARATSUBA_THRESHOLD)
	{
	  impn_sqr_n_basecase (prodp, up, size);
	}
      else
	{
	  mp_ptr tspace;
	  tspace = (mp_ptr) TMP_ALLOC (2 * size * BYTES_PER_MP_LIMB);
	  impn_sqr_n (prodp, up, size, tspace);
	}
    }
  else
    {
      if (size < KARATSUBA_THRESHOLD)
	{
	  impn_mul_n_basecase (prodp, up, vp, size);
	}
      else
	{
	  mp_ptr tspace;
	  tspace = (mp_ptr) TMP_ALLOC (2 * size * BYTES_PER_MP_LIMB);
	  impn_mul_n (prodp, up, vp, size, tspace);
	}
    }
  TMP_FREE (marker);
}
