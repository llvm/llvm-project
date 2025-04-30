/* Multiple versions of lroundf.
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

#include <math.h>
#include "init-arch.h"
#include <libm-alias-float.h>

/* It's safe to use double-precision implementation for single-precision.  */
extern __typeof (__lroundf) __lround_ppc32 attribute_hidden;
extern __typeof (__lroundf) __lround_power5plus attribute_hidden;
extern __typeof (__lroundf) __lround_power6x attribute_hidden;

libc_ifunc (__lroundf,
	    (hwcap & PPC_FEATURE_POWER6_EXT) ?
	      __lround_power6x
		: (hwcap & PPC_FEATURE_POWER5_PLUS) ?
		  __lround_power5plus
            : __lround_ppc32);

libm_alias_float (__lround, lround)
