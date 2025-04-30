/* Total order operation on absolute values.  ldbl-128ibm version.
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
#include <nan-high-order-bit.h>
#include <stdint.h>
#include <shlib-compat.h>

int
__totalordermagl (const long double *x, const long double *y)
{
  double xhi, xlo, yhi, ylo;
  int64_t hx, hy, lx, ly;

  ldbl_unpack (*x, &xhi, &xlo);
  EXTRACT_WORDS64 (hx, xhi);
  ldbl_unpack (*y, &yhi, &ylo);
  EXTRACT_WORDS64 (hy, yhi);
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
# error not implemented
#endif
  uint64_t x_sign = hx & 0x8000000000000000ULL;
  uint64_t y_sign = hy & 0x8000000000000000ULL;
  hx ^= x_sign;
  hy ^= y_sign;
  if (hx < hy)
    return 1;
  else if (hx > hy)
    return 0;

  /* The high doubles are identical.  If they are NaNs or both the low
     parts are zero, the low parts are not significant (and if they
     are infinities, both the low parts must be zero).  */
  if (hx >= 0x7ff0000000000000ULL)
    return 1;
  EXTRACT_WORDS64 (lx, xlo);
  EXTRACT_WORDS64 (ly, ylo);
  if (((lx | ly) & 0x7fffffffffffffffULL) == 0)
    return 1;
  lx ^= x_sign;
  ly ^= y_sign;

  /* Otherwise compare the low parts.  */
  uint64_t lx_sign = lx >> 63;
  uint64_t ly_sign = ly >> 63;
  lx ^= lx_sign >> 1;
  ly ^= ly_sign >> 1;
  return lx <= ly;
}
versioned_symbol (libm, __totalordermagl, totalordermagl, GLIBC_2_31);
#if SHLIB_COMPAT (libm, GLIBC_2_25, GLIBC_2_31)
int
attribute_compat_text_section
__totalordermag_compatl (long double x, long double y)
{
  return __totalordermagl (&x, &y);
}
compat_symbol (libm, __totalordermag_compatl, totalordermagl, GLIBC_2_25);
#endif
