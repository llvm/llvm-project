/* Multiple versions of llrintf.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
/* Redefine lrintf/__lrintf so that the compiler won't complain about the type
   mismatch with the IFUNC selector in strong_alias below.  */
#define lrintf __hidden_lrintf
#define __lrintf __hidden___lrintf

#include <math.h>
#undef lrintf
#undef __lrintf
#include "init-arch.h"
#include <libm-alias-float.h>

extern __typeof (__llrintf) __llrint_ppc64 attribute_hidden;
extern __typeof (__llrintf) __llrint_power6x attribute_hidden;
extern __typeof (__llrintf) __llrint_power8 attribute_hidden;

/* The ppc64 ABI passes float and double parameters in 64bit floating point
   registers (at least up to a point) as IEEE binary64 format, so effectively
   of "double" type.  Both l[l]rint and l[l]rintf return long type.  So these
   functions have identical signatures and functionality, and can use a
   single implementation.  */
libc_ifunc (__llrintf,
	    (hwcap2 & PPC_FEATURE2_ARCH_2_07)
	    ? __llrint_power8 :
	    (hwcap & PPC_FEATURE_POWER6_EXT)
	    ? __llrint_power6x
	    : __llrint_ppc64);

libm_alias_float (__llrint, llrint)
strong_alias (__llrintf, __lrintf)
libm_alias_float (__lrint, lrint)
