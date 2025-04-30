/* Test whether long double value is canonical.  ldbl-128ibm version.
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

#include <math.h>
#include <math_private.h>
#include <stdint.h>

int
__iscanonicall (long double x)
{
  double xhi, xlo;
  uint64_t hx, lx;

  ldbl_unpack (x, &xhi, &xlo);
  EXTRACT_WORDS64 (hx, xhi);
  EXTRACT_WORDS64 (lx, xlo);
  int64_t ix = hx & 0x7fffffffffffffffULL;
  int64_t iy = lx & 0x7fffffffffffffffULL;
  int hexp = (ix & 0x7ff0000000000000LL) >> 52;
  int lexp = (iy & 0x7ff0000000000000LL) >> 52;

  if (iy == 0)
    /* Low part 0 is always OK.  */
    return 1;

  if (hexp == 0x7ff)
    /* If a NaN, the low part does not matter.  If an infinity, the
       low part must be 0, in which case we have already returned.  */
    return ix != 0x7ff0000000000000LL;

  /* The high part is finite and the low part is nonzero.  There must
     be sufficient difference between the exponents.  */
  bool low_p2;
  if (lexp == 0)
    {
      /* Adjust the exponent for subnormal low part.  */
      lexp = 12 - __builtin_clzll (iy);
      low_p2 = iy == (1LL << (51 + lexp));
    }
  else
    low_p2 = (iy & 0xfffffffffffffLL) == 0;
  int expdiff = hexp - lexp;
  return expdiff > 53 || (expdiff == 53 && low_p2 && (ix & 1) == 0);
}
libm_hidden_def (__iscanonicall)
