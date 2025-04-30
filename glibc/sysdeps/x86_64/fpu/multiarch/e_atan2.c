/* Multiple versions of IEEE 754 atan.
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

#include <libm-alias-finite.h>

extern double __redirect_ieee754_atan2 (double, double);

#define SYMBOL_NAME ieee754_atan2
#include "ifunc-avx-fma4.h"

libc_ifunc_redirected (__redirect_ieee754_atan2,
		       __ieee754_atan2, IFUNC_SELECTOR ());
libm_alias_finite (__ieee754_atan2, __atan2)

#define __ieee754_atan2 __ieee754_atan2_sse2
#include <sysdeps/ieee754/dbl-64/e_atan2.c>
