/* Compute x^2 + y^2 - 1, without large cancellation error.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <math_private.h>
#include <fenv_private.h>
#include <mul_split.h>
#include <stdlib.h>

/* Calculate X + Y exactly and store the result in *HI + *LO.  It is
   given that |X| >= |Y| and the values are small enough that no
   overflow occurs.  */

static inline void
add_split (double *hi, double *lo, double x, double y)
{
  /* Apply Dekker's algorithm.  */
  *hi = x + y;
  *lo = (x - *hi) + y;
}

/* Compare absolute values of floating-point values pointed to by P
   and Q for qsort.  */

static int
compare (const void *p, const void *q)
{
  double pd = fabs (*(const double *) p);
  double qd = fabs (*(const double *) q);
  if (pd < qd)
    return -1;
  else if (pd == qd)
    return 0;
  else
    return 1;
}

/* Return X^2 + Y^2 - 1, computed without large cancellation error.
   It is given that 1 > X >= Y >= epsilon / 2, and that X^2 + Y^2 >=
   0.5.  */

long double
__x2y2m1l (long double x, long double y)
{
  double vals[13];
  SET_RESTORE_ROUND (FE_TONEAREST);
  union ibm_extended_long_double xu, yu;
  xu.ld = x;
  yu.ld = y;
  if (fabs (xu.d[1].d) < 0x1p-500)
    xu.d[1].d = 0.0;
  if (fabs (yu.d[1].d) < 0x1p-500)
    yu.d[1].d = 0.0;
  mul_split (&vals[1], &vals[0], xu.d[0].d, xu.d[0].d);
  mul_split (&vals[3], &vals[2], xu.d[0].d, xu.d[1].d);
  vals[2] *= 2.0;
  vals[3] *= 2.0;
  mul_split (&vals[5], &vals[4], xu.d[1].d, xu.d[1].d);
  mul_split (&vals[7], &vals[6], yu.d[0].d, yu.d[0].d);
  mul_split (&vals[9], &vals[8], yu.d[0].d, yu.d[1].d);
  vals[8] *= 2.0;
  vals[9] *= 2.0;
  mul_split (&vals[11], &vals[10], yu.d[1].d, yu.d[1].d);
  vals[12] = -1.0;
  qsort (vals, 13, sizeof (double), compare);
  /* Add up the values so that each element of VALS has absolute value
     at most equal to the last set bit of the next nonzero
     element.  */
  for (size_t i = 0; i <= 11; i++)
    {
      add_split (&vals[i + 1], &vals[i], vals[i + 1], vals[i]);
      qsort (vals + i + 1, 12 - i, sizeof (double), compare);
    }
  /* Now any error from this addition will be small.  */
  long double retval = (long double) vals[12];
  for (size_t i = 11; i != (size_t) -1; i--)
    retval += (long double) vals[i];
  return retval;
}
