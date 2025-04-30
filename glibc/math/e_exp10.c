/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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
#include <math_private.h>
#include <float.h>
#include <libm-alias-finite.h>

double
__ieee754_exp10 (double arg)
{
  if (isfinite (arg) && arg < DBL_MIN_10_EXP - DBL_DIG - 10)
    return DBL_MIN * DBL_MIN;
  else
    /* This is a very stupid and inprecise implementation.  It'll get
       replaced sometime (soon?).  */
    return __ieee754_exp (M_LN10 * arg);
}
libm_alias_finite (__ieee754_exp10, __exp10)
