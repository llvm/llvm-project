/* Multiple versions of ceilf.
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

#define NO_MATH_REDIRECT
#include <math.h>
#include <libm-alias-float.h>
#include "init-arch.h"

extern __typeof (__ceilf) __ceilf_ppc64 attribute_hidden;
extern __typeof (__ceilf) __ceilf_power5plus attribute_hidden;

libc_ifunc (__ceilf,
	    (hwcap & PPC_FEATURE_POWER5_PLUS)
	    ? __ceilf_power5plus
            : __ceilf_ppc64);

libm_alias_float (__ceil, ceil)
