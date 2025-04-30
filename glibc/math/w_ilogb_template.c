/* Wrapper to set errno for ilogb.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <errno.h>
#include <limits.h>
#include <math_private.h>
#include <fenv.h>

/* wrapper ilogb */
int
M_DECL_FUNC (__ilogb) (FLOAT x)
{
  int r = M_SUF (__ieee754_ilogb) (x);
  if (__builtin_expect (r == FP_ILOGB0, 0)
      || __builtin_expect (r == FP_ILOGBNAN, 0)
      || __builtin_expect (r == INT_MAX, 0))
    {
      __set_errno (EDOM);
      __feraiseexcept (FE_INVALID);
    }
  return r;
}
declare_mgen_alias (__ilogb, ilogb)
