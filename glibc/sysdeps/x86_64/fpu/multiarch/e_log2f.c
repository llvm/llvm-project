/* Multiple versions of log2f.
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

extern float __redirect_log2f (float);

#define SYMBOL_NAME log2f
#include "ifunc-fma.h"

libc_ifunc_redirected (__redirect_log2f, __log2f, IFUNC_SELECTOR ());

#ifdef SHARED
__hidden_ver1 (__log2f, __GI___log2f, __redirect_log2f)
  __attribute__ ((visibility ("hidden")));

versioned_symbol (libm, __ieee754_log2f, log2f, GLIBC_2_27);
libm_alias_float_other (__log2, log2)
#else
libm_alias_float (__log2, log2)
#endif

strong_alias (__log2f, __ieee754_log2f)
libm_alias_finite (__log2f, __log2f)

#define __log2f __log2f_sse2
#include <sysdeps/ieee754/flt-32/e_log2f.c>
