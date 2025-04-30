/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   Contributed by David Mosberger (davidm@cs.arizona.edu).
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <shlib-compat.h>

#include <sysdeps/ieee754/dbl-64/e_sqrt.c>

#if SHLIB_COMPAT (libm, GLIBC_2_18, GLIBC_2_31)
strong_alias (__ieee754_sqrt, __sqrt_finite_2_18)
compat_symbol (libm, __sqrt_finite_2_18, __sqrt_finite, GLIBC_2_18);
#endif
