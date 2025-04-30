/* Common definitions for libm tests for long double arguments to
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

#include <float.h>

#define ARG_FUNC(function) function ## l
#define ARG_FLOAT long double
#define ARG_PREFIX LDBL
#define ARG_LIT(x) (x ## L)
#if LDBL_MANT_DIG == DBL_MANT_DIG
# define ARG_TYPE_STR "double"
#else
# define ARG_TYPE_STR "ldouble"
#endif
#define FUNC_NARROW_SUFFIX l
