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

#define __llroundf	not___llroundf
#define llroundf	not_llroundf
#include <math.h>
#include <libm-alias-float.h>
#undef __llroundf
#undef llroundf


long int
__lroundf (float x)
{
  float adj, y;

  adj = copysignf (0.5f, x);
  asm("adds/suc %1,%2,%0" : "=&f"(y) : "f"(x), "f"(adj));
  return y;
}

strong_alias (__lroundf, __llroundf)
libm_alias_float (__lround, lround)
libm_alias_float (__llround, llround)
