/* Multiple versions of exp2f.
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

#include <libm-alias-float.h>
#include <libm-alias-finite.h>

extern float __redirect_exp2f (float);

#define SYMBOL_NAME exp2f
#include "ifunc-fma.h"

libc_ifunc_redirected (__redirect_exp2f, __exp2f, IFUNC_SELECTOR ());

#ifdef SHARED
versioned_symbol (libm, __ieee754_exp2f, exp2f, GLIBC_2_27);
libm_alias_float_other (__exp2, exp2)
#else
libm_alias_float (__exp2, exp2)
#endif

strong_alias (__exp2f, __ieee754_exp2f)
libm_alias_finite (__exp2f, __exp2f)

#define __exp2f __exp2f_sse2
#include <sysdeps/ieee754/flt-32/e_exp2f.c>
