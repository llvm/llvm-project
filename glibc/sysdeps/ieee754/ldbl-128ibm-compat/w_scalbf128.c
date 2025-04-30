/* Multiply _Float128 by integral power of 2
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#include <float128_private.h>
#include <math-type-macros-float128.h>

#undef declare_mgen_alias
#define declare_mgen_alias(a,b)
#define __ieee754_scalbl __ieee754_scalbf128
#include <w_scalb_template.c>

libm_alias_float128_other_r_ldbl (__scalb, scalb,)
