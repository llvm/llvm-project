/* Private inline math functions for x86.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#ifndef X86_MATH_PRIVATE_H
#define X86_MATH_PRIVATE_H 1

#include_next <math_private.h>

__extern_always_inline long double
__NTH (__ieee754_atan2l (long double y, long double x))
{
  long double ret;
  __asm__ __volatile__ ("fpatan" : "=t" (ret) : "0" (x), "u" (y) : "st(1)");
  return ret;
}

#endif
