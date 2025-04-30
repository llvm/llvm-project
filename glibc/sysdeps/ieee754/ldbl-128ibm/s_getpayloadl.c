/* Get NaN payload.  ldbl-128ibm version.
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

#include <fix-int-fp-convert-zero.h>
#include <math.h>
#include <math_private.h>
#include <stdint.h>

long double
__getpayloadl (const long double *x)
{
  double xhi = ldbl_high (*x);
  uint64_t ix;
  EXTRACT_WORDS64 (ix, xhi);
  if ((ix & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL
      || (ix & 0xfffffffffffffULL) == 0)
    return -1;
  ix &= 0x7ffffffffffffULL;
  if (FIX_INT_FP_CONVERT_ZERO && ix == 0)
    return 0.0L;
  return (long double) ix;
}
weak_alias (__getpayloadl, getpayloadl)
