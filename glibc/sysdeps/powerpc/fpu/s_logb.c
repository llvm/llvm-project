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
# include <sysdeps/ieee754/dbl-64/s_logb.c>
#else
# include <math.h>
# include <math_private.h>
# include <math_ldbl_opt.h>
# include <libm-alias-double.h>

/* This implementation avoids FP to INT conversions by using VSX
   bitwise instructions over FP values.  */
double
__logb (double x)
{
  double ret;

  if (__glibc_unlikely (x == 0.0))
    /* Raise FE_DIVBYZERO and return -HUGE_VAL[LF].  */
    return -1.0 / fabs (x);

  /* Mask to extract the exponent.  */
  asm ("xxland %x0,%x1,%x2\n"
       "fcfid  %0,%0"
       : "=d" (ret)
       : "d" (x), "d" (0x7ff0000000000000ULL));
  ret = (ret * 0x1p-52) - 1023.0;
  if (ret > 1023.0)
    /* Multiplication is used to set logb (+-INF) = INF.  */
    return (x * x);
  else if (ret == -1023.0)
    {
      /* POSIX specifies that denormal numbers are treated as
         though they were normalized.  */
      int64_t ix;
      EXTRACT_WORDS64 (ix, x);
      ix &= UINT64_C (0x7fffffffffffffff);
      return (double) (-1023 - (__builtin_clzll (ix) - 12));
    }
  /* Test to avoid logb_downward (0.0) == -0.0.  */
  return ret == -0.0 ? 0.0 : ret;
}
# ifndef __logb
libm_alias_double (__logb, logb)
# endif
#endif
