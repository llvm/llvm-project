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

#define __llrint	not___llrint
#define llrint		not_llrint
#include <math.h>
#include <math_ldbl_opt.h>
#include <libm-alias-double.h>
#undef __llrint
#undef llrint

long int
__lrint (double x)
{
  long ret;

  __asm ("cvttq/svd %1,%0" : "=&f"(ret) : "f"(x));

  return ret;
}

strong_alias (__lrint, __llrint)
libm_alias_double (__lrint, lrint)
libm_alias_double (__llrint, llrint)
