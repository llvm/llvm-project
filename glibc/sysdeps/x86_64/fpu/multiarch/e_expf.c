/* Multiple versions of expf.
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

extern float __redirect_expf (float);

#define SYMBOL_NAME expf
#include "ifunc-fma.h"

libc_ifunc_redirected (__redirect_expf, __expf, IFUNC_SELECTOR ());

#ifdef SHARED
__hidden_ver1 (__expf, __GI___expf, __redirect_expf)
  __attribute__ ((visibility ("hidden")));

versioned_symbol (libm, __ieee754_expf, expf, GLIBC_2_27);
libm_alias_float_other (__exp, exp)
#else
libm_alias_float (__exp, exp)
#endif

strong_alias (__expf, __ieee754_expf)
libm_alias_finite (__expf, __expf)

#define __expf __expf_sse2
#include <sysdeps/ieee754/flt-32/e_expf.c>
