/* mpn_submul_1 -- multiply the S1_SIZE long limb vector pointed to by S1_PTR
   by S2_LIMB, subtract the S1_SIZE least significant limbs of the product
   from the limb vector pointed to by RES_PTR.  Return the most significant
   limb of the product, adjusted for carry-out from the subtraction.

Copyright (C) 1992-2021 Free Software Foundation, Inc.

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

mp_limb_t
mpn_submul_1 (register mp_ptr res_ptr, register mp_srcptr s1_ptr,
	      mp_size_t s1_size, register mp_limb_t s2_limb)
{
  register mp_limb_t cy_limb;
  register mp_size_t j;
  register mp_limb_t prod_high, prod_low;
  register mp_limb_t x;

  /* The loop counter and index J goes from -SIZE to -1.  This way
     the loop becomes faster.  */
  j = -s1_size;

  /* Offset the base pointers to compensate for the negative indices.  */
  res_ptr -= j;
  s1_ptr -= j;

  cy_limb = 0;
  do
    {
      umul_ppmm (prod_high, prod_low, s1_ptr[j], s2_limb);

      prod_low += cy_limb;
      cy_limb = (prod_low < cy_limb) + prod_high;

      x = res_ptr[j];
      prod_low = x - prod_low;
      cy_limb += (prod_low > x);
      res_ptr[j] = prod_low;
    }
  while (++j != 0);

  return cy_limb;
}
