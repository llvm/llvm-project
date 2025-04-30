/* Auxiliary routine for the Bessel functions (j0f, y0f, j1f, y1f).
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _MATH_REDUCE_AUX_H
#define _MATH_REDUCE_AUX_H

#include <math.h>
#include <math_private.h>
#include <s_sincosf.h>

/* Return h and update n such that:
   Now x - pi/4 - alpha = h + n*pi/2 mod (2*pi).  */
static inline double
reduce_aux (float x, int *n, double alpha)
{
  double h;
  h = reduce_large (asuint (x), n);
  /* Now |x| = h+n*pi/2 mod 2*pi.  */
  /* Recover sign.  */
  if (x < 0)
    {
      h = -h;
      *n = -*n;
    }
  /* Subtract pi/4.  */
  double piover2 = 0xc.90fdaa22168cp-3;
  if (h >= 0)
    h -= piover2 / 2;
  else
    {
      h += piover2 / 2;
      (*n) --;
    }
  /* Subtract alpha and reduce if needed mod pi/2.  */
  h -= alpha;
  if (h > piover2)
    {
      h -= piover2;
      (*n) ++;
    }
  else if (h < -piover2)
    {
      h += piover2;
      (*n) --;
    }
  return h;
}

#endif
