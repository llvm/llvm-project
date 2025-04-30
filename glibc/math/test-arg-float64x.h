/* Common definitions for libm tests for _Float64x arguments to
   narrowing functions.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <bits/floatn.h>
#include <float.h>

#define ARG_FUNC(function) function ## f64x
#define ARG_FLOAT _Float64x
#define ARG_PREFIX FLT64X
#define ARG_LIT(x) __f64x (x)
#if FLT64X_MANT_DIG == LDBL_MANT_DIG
# define ARG_TYPE_STR "ldouble"
#else
# define ARG_TYPE_STR "float128"
#endif
#define FUNC_NARROW_SUFFIX f64x
