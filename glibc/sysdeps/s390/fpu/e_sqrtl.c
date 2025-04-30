/* Square root.  S/390 FPU version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky <schwidefsky@de.ibm.com>.
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

#include <math_private.h>
#include <libm-alias-finite.h>

long double
__ieee754_sqrtl (long double x)
{
  long double res;

  __asm__ ( "sqxbr %0,%1" : "=f" (res) : "f" (x) );
  return res;
}
libm_alias_finite (__ieee754_sqrtl, __sqrtl)
