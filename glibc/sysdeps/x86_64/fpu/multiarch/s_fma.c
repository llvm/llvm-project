/* FMA version of fma.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
   Contributed by Intel Corporation.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <config.h>
#include <math.h>
#include <init-arch.h>
#include <libm-alias-double.h>

extern double __fma_sse2 (double x, double y, double z) attribute_hidden;


static double
__fma_fma3 (double x, double y, double z)
{
  asm ("vfmadd213sd %3, %2, %0" : "=x" (x) : "0" (x), "x" (y), "xm" (z));
  return x;
}


static double
__fma_fma4 (double x, double y, double z)
{
  asm ("vfmaddsd %3, %2, %1, %0" : "=x" (x) : "x" (x), "x" (y), "x" (z));
  return x;
}


libm_ifunc (__fma, CPU_FEATURE_USABLE (FMA)
	    ? __fma_fma3 : (CPU_FEATURE_USABLE (FMA4)
			    ? __fma_fma4 : __fma_sse2));
libm_alias_double (__fma, fma)

#define __fma __fma_sse2

#include <sysdeps/ieee754/dbl-64/s_fma.c>
