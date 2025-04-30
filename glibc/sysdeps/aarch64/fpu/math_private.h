/* Configure optimized libm functions.  AArch64 version.
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

#ifndef AARCH64_MATH_PRIVATE_H
#define AARCH64_MATH_PRIVATE_H 1

#include <stdint.h>
#include <math.h>

/* Use inline round and lround instructions.  */
#define TOINT_INTRINSICS 1

static inline double_t
roundtoint (double_t x)
{
  return round (x);
}

static inline int32_t
converttoint (double_t x)
{
  return lround (x);
}

#include_next <math_private.h>

#endif
