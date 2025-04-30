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
#include <math_ldbl_opt.h>

/* Return the least floating-point number greater than X.  */
long double
__nextupl (long double x)
{
  int64_t hx, ihx, lx;
  double xhi, xlo, yhi;

  ldbl_unpack (x, &xhi, &xlo);
  EXTRACT_WORDS64 (hx, xhi);
  EXTRACT_WORDS64 (lx, xlo);
  ihx = hx & 0x7fffffffffffffffLL;

  if (ihx > 0x7ff0000000000000LL)	/* x is nan.  */
    return x + x;		/* Signal the nan.  */
  if (ihx == 0)
    return LDBL_TRUE_MIN;

  long double u;
  if ((hx == 0x7fefffffffffffffLL) && (lx == 0x7c8ffffffffffffeLL))
    return INFINITY;
  if ((uint64_t) hx >= 0xfff0000000000000ULL)
    {
      u = -0x1.fffffffffffff7ffffffffffff8p+1023L;
      return u;
    }
  if (ihx <= 0x0360000000000000LL)
    {				/* x <= LDBL_MIN.  */
      x += LDBL_TRUE_MIN;
      if (x == 0.0L)		/* Handle negative LDBL_TRUE_MIN case.  */
        x = -0.0L;
      return x;
    }
  /* If the high double is an exact power of two and the low
     double is the opposite sign, then 1ulp is one less than
     what we might determine from the high double.  Similarly
     if X is an exact power of two, and negative, because
     making it a little larger will result in the exponent
     decreasing by one and normalisation of the mantissa.   */
  if ((hx & 0x000fffffffffffffLL) == 0
      && ((lx != 0 && lx != 0x8000000000000000LL && (hx ^ lx) < 0)
          || ((lx == 0 || lx == 0x8000000000000000LL) && hx < 0)))
    ihx -= 1LL << 52;
  if (ihx < (106LL << 52))
    {				/* ulp will denormal.  */
      INSERT_WORDS64 (yhi, ihx & (0x7ffLL << 52));
      u = yhi * 0x1p-105;
    }
  else
    {
      INSERT_WORDS64 (yhi, (ihx & (0x7ffLL << 52)) - (105LL << 52));
      u = yhi;
    }
  return x + u;
}

weak_alias (__nextupl, nextupl)
