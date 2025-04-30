/* Smallest integral value not less than argument.  PowerPC version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#define NO_MATH_REDIRECT
#include <math.h>
#include <libm-alias-double.h>
#include <round_to_integer.h>

double
__ceil (double x)
{
#ifdef _ARCH_PWR5X
  return __builtin_ceil (x);
#else
  return round_to_integer_double (CEIL, x);
#endif
}
#ifndef __ceil
libm_alias_double (__ceil, ceil)
#endif
