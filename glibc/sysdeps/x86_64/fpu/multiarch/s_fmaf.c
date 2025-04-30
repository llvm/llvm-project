/* FMA version of fmaf.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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
#include <libm-alias-float.h>

extern float __fmaf_sse2 (float x, float y, float z) attribute_hidden;


static float
__fmaf_fma3 (float x, float y, float z)
{
  asm ("vfmadd213ss %3, %2, %0" : "=x" (x) : "0" (x), "x" (y), "xm" (z));
  return x;
}


static float
__fmaf_fma4 (float x, float y, float z)
{
  asm ("vfmaddss %3, %2, %1, %0" : "=x" (x) : "x" (x), "x" (y), "x" (z));
  return x;
}


libm_ifunc (__fmaf, CPU_FEATURE_USABLE (FMA)
	    ? __fmaf_fma3 : (CPU_FEATURE_USABLE (FMA4)
			     ? __fmaf_fma4 : __fmaf_sse2));
libm_alias_float (__fma, fma)

#define __fmaf __fmaf_sse2

#include <sysdeps/ieee754/dbl-64/s_fmaf.c>
