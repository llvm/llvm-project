/* Multiple versions of lrint.
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
#include <math_ldbl_opt.h>
#include <shlib-compat.h>
#include "init-arch.h"
#include <libm-alias-double.h>

extern __typeof (__lrint) __lrint_ppc32 attribute_hidden;
extern __typeof (__lrint) __lrint_power6x attribute_hidden;

libc_ifunc (__lrint,
	    (hwcap & PPC_FEATURE_POWER6_EXT) ?
	      __lrint_power6x
            : __lrint_ppc32);

libm_alias_double (__lrint, lrint)
