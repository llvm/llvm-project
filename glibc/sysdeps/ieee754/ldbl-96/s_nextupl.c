/* Return the least floating-point number greater than X.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>

/* Return the least floating-point number greater than X.  */
long double
__nextupl (long double x)
{
  uint32_t hx, ix;
  uint32_t lx;
  int32_t esx;

  GET_LDOUBLE_WORDS (esx, hx, lx, x);
  ix = esx & 0x7fff;

  if (((ix == 0x7fff) && (((hx & 0x7fffffff) | lx) != 0)))  /* x is nan.  */
    return x + x;
  if ((ix | hx | lx) == 0)
    return LDBL_TRUE_MIN;
  if (esx >= 0)
    {				/* x > 0.  */
      if (isinf (x))
        return x;
      lx += 1;
      if (lx == 0)
        {
          hx += 1;
#if LDBL_MIN_EXP == -16381
          if (hx == 0 || (esx == 0 && hx == 0x80000000))
#else
          if (hx == 0)
#endif
          {
            esx += 1;
            hx |= 0x80000000;
          }
        }
    }
  else
    {				/* x < 0.  */
      if (lx == 0)
        {
#if LDBL_MIN_EXP == -16381
          if (hx <= 0x80000000 && esx != 0xffff8000)
            {
              esx -= 1;
              hx = hx - 1;
              if ((esx & 0x7fff) > 0)
                hx |= 0x80000000;
            }
          else
            hx -= 1;
#else
          if (ix != 0 && hx == 0x80000000)
            hx = 0;
          if (hx == 0)
            esx -= 1;
          hx -= 1;
#endif
        }
      lx -= 1;
    }
  SET_LDOUBLE_WORDS (x, esx, hx, lx);
  return x;
}

libm_alias_ldouble (__nextup, nextup)
