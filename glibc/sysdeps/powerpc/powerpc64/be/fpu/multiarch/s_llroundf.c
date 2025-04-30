/* Multiple versions of llroundf.
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
/* Redefine lroundf/__lroundf so that the compiler won't complain about
   the type mismatch with the IFUNC selector in strong_alias below.  */
#define lroundf __hidden_lroundf
#define __lroundf __hidden___lroundf

#include <math.h>
#undef lroundf
#undef __lroundf
#include "init-arch.h"
#include <libm-alias-float.h>

extern __typeof (__llroundf) __llroundf_ppc64 attribute_hidden;
extern __typeof (__llroundf) __llround_power6x attribute_hidden;
extern __typeof (__llroundf) __llround_power8 attribute_hidden;

/* The ppc64 ABI passes float and double parameters in 64bit floating point
   registers (at least up to a point) as IEEE binary64 format, so effectively
   of "double" type.  Both l[l]round and l[l]roundf return long type.  So these
   functions have identical signatures and functionality, and can use a
   single implementation.  */
libc_ifunc (__llroundf,
	    (hwcap2 & PPC_FEATURE2_ARCH_2_07)
	    ? __llround_power8 :
	    (hwcap & PPC_FEATURE_POWER6_EXT)
	    ? __llround_power6x
	    : __llroundf_ppc64);

libm_alias_float (__llround, llround)
strong_alias (__llroundf, __lroundf)
libm_alias_float (__lround, lround)
