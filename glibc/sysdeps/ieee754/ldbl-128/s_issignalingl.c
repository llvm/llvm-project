/* Test for signaling NaN.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

int
__issignalingl (_Float128 x)
{
  uint64_t hxi, lxi __attribute__ ((unused));
  GET_LDOUBLE_WORDS64 (hxi, lxi, x);
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
  /* We only have to care about the high-order bit of x's significand, because
     having it set (sNaN) already makes the significand different from that
     used to designate infinity.  */
  return ((hxi & UINT64_C (0x7fff800000000000))
          == UINT64_C (0x7fff800000000000));
#else
  /* To keep the following comparison simple, toggle the quiet/signaling bit,
     so that it is set for sNaNs.  This is inverse to IEEE 754-2008 (as well as
     common practice for IEEE 754-1985).  */
  hxi ^= UINT64_C (0x0000800000000000);
  /* If lxi != 0, then set any suitable bit of the significand in hxi.  */
  hxi |= (lxi | -lxi) >> 63;
  /* We have to compare for greater (instead of greater or equal), because x's
     significand being all-zero designates infinity not NaN.  */
  return (hxi & UINT64_C (0x7fffffffffffffff)) > UINT64_C (0x7fff800000000000);
#endif
}
libm_hidden_def (__issignalingl)
