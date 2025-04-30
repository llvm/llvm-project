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
#include <mul_splitl.h>
#include <stdlib.h>

/* Calculate X + Y exactly and store the result in *HI + *LO.  It is
   given that |X| >= |Y| and the values are small enough that no
   overflow occurs.  */

static inline void
add_split (long double *hi, long double *lo, long double x, long double y)
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
  long double pld = fabsl (*(const long double *) p);
  long double qld = fabsl (*(const long double *) q);
  if (pld < qld)
    return -1;
  else if (pld == qld)
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
  long double vals[5];
  SET_RESTORE_ROUNDL (FE_TONEAREST);
  mul_splitl (&vals[1], &vals[0], x, x);
  mul_splitl (&vals[3], &vals[2], y, y);
  vals[4] = -1.0L;
  qsort (vals, 5, sizeof (long double), compare);
  /* Add up the values so that each element of VALS has absolute value
     at most equal to the last set bit of the next nonzero
     element.  */
  for (size_t i = 0; i <= 3; i++)
    {
      add_split (&vals[i + 1], &vals[i], vals[i + 1], vals[i]);
      qsort (vals + i + 1, 4 - i, sizeof (long double), compare);
    }
  /* Now any error from this addition will be small.  */
  return vals[4] + vals[3] + vals[2] + vals[1] + vals[0];
}
