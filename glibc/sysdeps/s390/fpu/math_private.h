/* Configure optimized libm functions.  S390 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifndef S390_MATH_PRIVATE_H
#define S390_MATH_PRIVATE_H 1

#include <stdint.h>
#include <math.h>

#ifdef HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
# define TOINT_INTRINSICS 1

static inline double_t
roundtoint (double_t x)
{
  double_t y;
  /* The z196 zarch "load fp integer" (fidbra) instruction is rounding
     x to the nearest integer with ties away from zero (M3-field: 1)
     where inexact exceptions are suppressed (M4-field: 4).  */
  __asm__ ("fidbra %0,1,%1,4" : "=f" (y) : "f" (x));
  return y;
}

static inline int32_t
converttoint (double_t x)
{
  int32_t y;
  /* The z196 zarch "convert to fixed" (cfdbra) instruction is rounding
     x to the nearest integer with ties away from zero (M3-field: 1)
     where inexact exceptions are suppressed (M4-field: 4).  */
  __asm__ ("cfdbra %0,1,%1,4" : "=d" (y) : "f" (x) : "cc");
  return y;
}
#endif

#include_next <math_private.h>

#endif
