/* llroundf() - S390 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#if defined __s390x__ && defined HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
/* We only support s390x as on s390 a long long int refers to a register pair
   of two 4byte registers instead of a 8byte register which is produced by the
   instruction.
   Note: On s390 this instruction would only be used if build with -mzarch.  */
# include <math.h>
# include <libm-alias-float.h>

long long int
__llroundf (float x)
{
  long long int y;
  /* The z196 zarch "convert to fixed" (cgebra) instruction is rounding
     x to the nearest integer with "ties away from 0" rounding mode
     (M3-field: 1) where inexact exceptions are suppressed (M4-field: 4).  */
  __asm__ ("cgebra %0,1,%1,4" : "=d" (y) : "f" (x) : "cc");
  return y;
}
libm_alias_float (__llround, llround)

#else
# include <sysdeps/ieee754/flt-32/s_llroundf.c>
#endif
