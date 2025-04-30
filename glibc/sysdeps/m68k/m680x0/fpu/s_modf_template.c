/* Implement modf for m68k.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include "mathimpl.h"

FLOAT
M_DECL_FUNC (__modf) (FLOAT x, FLOAT *iptr)
{
  FLOAT x_int, result;
  unsigned long x_cond;

  __asm ("fintrz%.x %1, %0" : "=f" (x_int) : "f" (x));
  *iptr = x_int;
  x_cond = __m81_test (x);
  if (x_cond & __M81_COND_INF)
    {
      result = 0;
      if (x_cond & __M81_COND_NEG)
	result = -result;
    }
  else if (x_cond & __M81_COND_ZERO)
    result = x;
  else
    result = x - x_int;
  return result;
}
declare_mgen_alias (__modf, modf)
