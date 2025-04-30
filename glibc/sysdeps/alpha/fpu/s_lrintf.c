/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#define __llrintf	not___llrintf
#define llrintf		not_llrintf
#include <math.h>
#include <libm-alias-float.h>
#undef __llrintf
#undef llrintf

long int
__lrintf (float x)
{
  double tmp;
  long ret;

  __asm ("cvtst/s %2,%1\n\tcvttq/svd %1,%0"
	 : "=&f"(ret), "=&f"(tmp) : "f"(x));

  return ret;
}

strong_alias (__lrintf, __llrintf)
libm_alias_float (__lrint, lrint)
libm_alias_float (__llrint, llrint)
