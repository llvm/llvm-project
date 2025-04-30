/* mpn_rshift -- Shift right a low-level natural-number integer.

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

#include <assert.h>

/* Shift U (pointed to by UP and USIZE limbs long) CNT bits to the right
   and store the USIZE least significant limbs of the result at WP.
   The bits shifted out to the right are returned.

   Argument constraints:
   1. 0 < CNT < BITS_PER_MP_LIMB
   2. If the result is to be written over the input, WP must be <= UP.
*/

mp_limb_t
mpn_rshift (register mp_ptr wp,
	    register mp_srcptr up, mp_size_t usize,
	    register unsigned int cnt)
{
  register mp_limb_t high_limb, low_limb;
  register unsigned sh_1, sh_2;
  register mp_size_t i;
  mp_limb_t retval;

  assert (usize != 0 && cnt != 0);

  sh_1 = cnt;

#if 0
  if (sh_1 == 0)
    {
      if (wp != up)
	{
	  /* Copy from low end to high end, to allow specified input/output
	     overlapping.  */
	  for (i = 0; i < usize; i++)
	    wp[i] = up[i];
	}
      return usize;
    }
#endif

  wp -= 1;
  sh_2 = BITS_PER_MP_LIMB - sh_1;
  high_limb = up[0];
  retval = high_limb << sh_2;
  low_limb = high_limb;

  for (i = 1; i < usize; i++)
    {
      high_limb = up[i];
      wp[i] = (low_limb >> sh_1) | (high_limb << sh_2);
      low_limb = high_limb;
    }
  wp[i] = low_limb >> sh_1;

  return retval;
}
