/* Total order operation.  ldbl-128ibm version.
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
__totalorderl (const long double *x, const long double *y)
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
  uint64_t hx_sign = hx >> 63;
  uint64_t hy_sign = hy >> 63;
  int64_t hx_adj = hx ^ (hx_sign >> 1);
  int64_t hy_adj = hy ^ (hy_sign >> 1);
  if (hx_adj < hy_adj)
    return 1;
  else if (hx_adj > hy_adj)
    return 0;

  /* The high doubles are identical.  If they are NaNs or both the low
     parts are zero, the low parts are not significant (and if they
     are infinities, both the low parts must be zero).  */
  if ((hx & 0x7fffffffffffffffULL) >= 0x7ff0000000000000ULL)
    return 1;
  EXTRACT_WORDS64 (lx, xlo);
  EXTRACT_WORDS64 (ly, ylo);
  if (((lx | ly) & 0x7fffffffffffffffULL) == 0)
    return 1;

  /* Otherwise compare the low parts.  */
  uint64_t lx_sign = lx >> 63;
  uint64_t ly_sign = ly >> 63;
  lx ^= lx_sign >> 1;
  ly ^= ly_sign >> 1;
  return lx <= ly;
}
versioned_symbol (libm, __totalorderl, totalorderl, GLIBC_2_31);
#if SHLIB_COMPAT (libm, GLIBC_2_25, GLIBC_2_31)
int
attribute_compat_text_section
__totalorder_compatl (long double x, long double y)
{
  return __totalorderl (&x, &y);
}
compat_symbol (libm, __totalorder_compatl, totalorderl, GLIBC_2_25);
#endif
