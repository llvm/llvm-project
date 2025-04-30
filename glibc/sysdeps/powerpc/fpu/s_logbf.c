/* Get exponent of a floating-point value.  PowerPC version.
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

/* ISA 2.07 provides fast GPR to FP instruction (mfvsr{d,wz}) which make
   generic implementation faster.  */
#if defined(_ARCH_PWR8) || !defined(_ARCH_PWR7)
# include <sysdeps/ieee754/flt-32/s_logbf.c>
#else
# include <math.h>
# include <libm-alias-float.h>
/* This implementation avoids FP to INT conversions by using VSX
   bitwise instructions over FP values.  */
float
__logbf (float x)
{
  /* VSX operation are all done internally as double.  */
  double ret;

  if (__glibc_unlikely (x == 0.0))
    /* Raise FE_DIVBYZERO and return -HUGE_VAL[LF].  */
    return -1.0 / fabs (x);

  /* mask to extract the exponent.  */
  asm ("xxland %x0,%x1,%x2\n"
       "fcfid  %0,%0"
       : "=d"(ret)
       : "d" (x), "d" (0x7ff0000000000000ULL));
  /* ret = (ret >> 52) - 1023.0, since ret is double.  */
  ret = (ret * 0x1p-52) - 1023.0;
  if (ret > 127.0)
    /* Multiplication is used to set logb (+-INF) = INF.  */
    return (x * x);
  /* Since operations are done with double we don't need
     additional tests for subnormal numbers.
     The test is to avoid logb_downward (0.0) == -0.0.  */
  return ret == -0.0 ? 0.0 : ret;
}
# ifndef __logbf
libm_alias_float (__logb, logb)
# endif
#endif
