/* llrint() - S390 version.
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
# include <libm-alias-double.h>

long long int
__llrint (double x)
{
  long long int y;
  /* The z196 zarch "convert to fixed" (cgdbra) instruction is rounding
     according to current rounding mode (M3-field: 0).
     First convert x with suppressed inexact exception and check if the
     resulting value is beyond the target limits (indicated by cc=3;
     Note: a nan is also indicated by cc=3).
     If the resulting value is within the target limits, redo
     without suppressing the inexact exception.  */
  __asm__ ("cgdbra %0,0,%1,4 \n\t"
	   "jo 1f \n\t"
	   "cgdbra %0,0,%1,0 \n\t"
	   "1:"
	   : "=&d" (y) : "f" (x) : "cc");
  return y;
}
libm_alias_double (__llrint, llrint)

#else
# include <sysdeps/ieee754/dbl-64/s_llrint.c>
#endif
