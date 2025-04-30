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

#define __llround	not___llround
#define llround		not_llround
#include <math.h>
#include <math_ldbl_opt.h>
#include <libm-alias-double.h>
#undef __llround
#undef llround

long int
__lround (double x)
{
  double adj, y;

  adj = copysign (0.5, x);
  asm("addt/suc %1,%2,%0" : "=&f"(y) : "f"(x), "f"(adj));
  return y;
}

strong_alias (__lround, __llround)
libm_alias_double (__lround, lround)
libm_alias_double (__llround, llround)
