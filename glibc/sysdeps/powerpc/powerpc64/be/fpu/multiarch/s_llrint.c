/* Multiple versions of llrint.
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

/* Redefine lrint/__lrint so that the compiler won't complain about the type
   mismatch with the IFUNC selector in strong_alias below.  */
#define lrint __hidden_lrint
#define __lrint __hidden___lrint

#include <math.h>
#include <math_ldbl_opt.h>
#undef lrint
#undef __lrint
#include <shlib-compat.h>
#include "init-arch.h"
#include <libm-alias-double.h>

extern __typeof (__llrint) __llrint_ppc64 attribute_hidden;
extern __typeof (__llrint) __llrint_power6x attribute_hidden;
extern __typeof (__llrint) __llrint_power8 attribute_hidden;

libc_ifunc (__llrint,
	    (hwcap2 & PPC_FEATURE2_ARCH_2_07)
	    ? __llrint_power8 :
	      (hwcap & PPC_FEATURE_POWER6_EXT)
	      ? __llrint_power6x
            : __llrint_ppc64);

libm_alias_double (__llrint, llrint)

/* long has the same width as long long on PowerPC64.  */
strong_alias (__llrint, __lrint)
libm_alias_double (__lrint, lrint)
