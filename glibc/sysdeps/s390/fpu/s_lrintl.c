/* lrintl() - S390 version.
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

#ifdef HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
# include <math.h>
# include <math_private.h>
# include <libm-alias-ldouble.h>

/* The sizeof (long int) differs between s390x (8byte) and s390 (4byte).
   Thus we need different instructions as the target size is encoded there.
   Note: On s390 this instruction is only used if build with -mzarch.  */
# ifdef __s390x__
#  define INSN "cgxbra"
# else
#  define INSN "cfxbra"
# endif

long int
__lrintl (_Float128 x)
{
  long int y;
  /* The z196 zarch "convert to fixed" (cgxbra) instruction is rounding
     according to current rounding mode (M3-field: 0).
     First convert x with suppressed inexact exception and check if the
     resulting value is beyond the target limits (indicated by cc=3;
     Note: a nan is also indicated by cc=3).
     If the resulting value is within the target limits, redo
     without suppressing the inexact exception.  */
  __asm__ (INSN " %0,0,%1,4 \n\t"
	   "jo 1f \n\t"
	   INSN " %0,0,%1,0 \n\t"
	   "1:"
	   : "=&d" (y) : "f" (x) : "cc");
  return y;
}
libm_alias_ldouble (__lrint, lrint)

#else
# include <sysdeps/ieee754/ldbl-128/s_lrintl.c>
#endif
