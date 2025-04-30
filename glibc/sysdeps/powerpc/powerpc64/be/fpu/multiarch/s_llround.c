/* Multiple versions of llround.
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

#define lround __hidden_lround
#define __lround __hidden___lround

#include <math.h>
#include <math_ldbl_opt.h>
#include <shlib-compat.h>
#include "init-arch.h"
#include <libm-alias-double.h>

extern __typeof (__llround) __llround_ppc64 attribute_hidden;
extern __typeof (__llround) __llround_power5plus attribute_hidden;
extern __typeof (__llround) __llround_power6x attribute_hidden;
extern __typeof (__llround) __llround_power8 attribute_hidden;

libc_ifunc (__llround,
	    (hwcap2 & PPC_FEATURE2_ARCH_2_07)
	    ? __llround_power8 :
	      (hwcap & PPC_FEATURE_POWER6_EXT)
	      ? __llround_power6x :
		(hwcap & PPC_FEATURE_POWER5_PLUS)
		? __llround_power5plus
            : __llround_ppc64);

libm_alias_double (__llround, llround)

/* long has the same width as long long on PPC64.  */
#undef lround
#undef __lround
strong_alias (__llround, __lround)
libm_alias_double (__lround, lround)
