/* mpn_add_n -- Add two limb vectors of equal, non-zero length.

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

mp_limb_t
mpn_add_n (mp_ptr res_ptr, mp_srcptr s1_ptr, mp_srcptr s2_ptr, mp_size_t size)
{
  register mp_limb_t x, y, cy;
  register mp_size_t j;

  /* The loop counter and index J goes from -SIZE to -1.  This way
     the loop becomes faster.  */
  j = -size;

  /* Offset the base pointers to compensate for the negative indices.  */
  s1_ptr -= j;
  s2_ptr -= j;
  res_ptr -= j;

  cy = 0;
  do
    {
      y = s2_ptr[j];
      x = s1_ptr[j];
      y += cy;			/* add previous carry to one addend */
      cy = (y < cy);		/* get out carry from that addition */
      y = x + y;		/* add other addend */
      cy = (y < x) + cy;	/* get out carry from that add, combine */
      res_ptr[j] = y;
    }
  while (++j != 0);

  return cy;
}
