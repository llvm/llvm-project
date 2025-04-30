/* Multiple versions of powf.
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

#define powf __redirect_powf
#define __DECL_SIMD___redirect_powf
#include <math.h>
#undef powf

#define SYMBOL_NAME powf
#include "ifunc-fma.h"

libc_ifunc_redirected (__redirect_powf, __powf, IFUNC_SELECTOR ());

#ifdef SHARED
__hidden_ver1 (__powf, __GI___powf, __redirect_powf)
  __attribute__ ((visibility ("hidden")));

versioned_symbol (libm, __ieee754_powf, powf, GLIBC_2_27);
libm_alias_float_other (__pow, pow)
#else
libm_alias_float (__pow, pow)
#endif

strong_alias (__powf, __ieee754_powf)
libm_alias_finite (__powf, __powf)

#define __powf __powf_sse2
#include <sysdeps/ieee754/flt-32/e_powf.c>
