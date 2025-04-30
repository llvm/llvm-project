/* Round to nearest integer.  PowerPC64 version.
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
#define lrint __redirect_lrint
#define lrintf __redirect_lrintf
#define llrintf __redirect_llrintf
#define __lrint __redirect___lrint
#define __lrintf __redirect___lrintf
#define __llrintf __redirect___llrintf
#include <math.h>
#undef lrint
#undef lrintf
#undef llrintf
#undef __lrint
#undef __lrintf
#undef __llrintf
#include <libm-alias-float.h>
#include <libm-alias-double.h>

long long int
__llrint (double x)
{
  long int ret;
  __asm__ ("fctid %0, %1" : "=d" (ret) : "d" (x));
  return ret;
}
#ifndef __llrint
strong_alias (__llrint, __lrint)
libm_alias_double (__llrint, llrint)
libm_alias_double (__lrint, lrint)

/* The double version also works for single-precision as both float and
   double parameters are passed in 64bit FPRs and both versions are expected
   to return [long] long type.  */
strong_alias (__llrint, __llrintf)
libm_alias_float (__llrint, llrint)
strong_alias (__lrint, __lrintf)
libm_alias_float (__lrint, lrint)
#endif
