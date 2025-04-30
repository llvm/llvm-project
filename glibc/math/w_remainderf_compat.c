/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>, 2011.

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
#include <math-svid-compat.h>
#include <libm-alias-float.h>


#if LIBM_SVID_COMPAT
/* wrapper remainderf */
float
__remainderf (float x, float y)
{
  if (((__builtin_expect (y == 0.0f, 0) && ! isnan (x))
       || (__builtin_expect (isinf (x), 0) && ! isnan (y)))
      && _LIB_VERSION != _IEEE_)
    return __kernel_standard_f (x, y, 128); /* remainder domain */

  return __ieee754_remainderf (x, y);
}
libm_alias_float (__remainder, remainder)
weak_alias (__remainderf, dremf)
#endif
