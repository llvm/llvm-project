/* Round to nearest integer.  PowerPC32 version.
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
#define lrintf __redirect_lrintf
#define __lrintf __redirect___lrintf
#include <math.h>
#undef lrintf
#undef __lrintf
#include <fenv_private.h>
#include <libm-alias-double.h>
#include <libm-alias-float.h>

long int
__lrint (double x)
{
  long long int ret;
  __asm__ ("fctiw %0, %1" : "=d" (ret) : "d" (x));
  return ret;
}
#ifndef __lrint
libm_alias_double (__lrint, lrint)
strong_alias (__lrint, __lrintf)
libm_alias_float (__lrint, lrint)
#endif
