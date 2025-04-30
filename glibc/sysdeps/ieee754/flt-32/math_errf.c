/* Single-precision math error handling.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include "math_config.h"

#if WANT_ERRNO
# include <errno.h>
/* NOINLINE reduces code size.  */
NOINLINE static float
with_errnof (float y, int e)
{
  errno = e;
  return y;
}
#else
# define with_errnof(x, e) (x)
#endif

/* NOINLINE prevents fenv semantics breaking optimizations.  */
NOINLINE static float
xflowf (uint32_t sign, float y)
{
  y = (sign ? -y : y) * y;
  return with_errnof (y, ERANGE);
}

attribute_hidden float
__math_uflowf (uint32_t sign)
{
  return xflowf (sign, 0x1p-95f);
}

#if WANT_ERRNO_UFLOW
/* Underflows to zero in some non-nearest rounding mode, setting errno
   is valid even if the result is non-zero, but in the subnormal range.  */
attribute_hidden float
__math_may_uflowf (uint32_t sign)
{
  return xflowf (sign, 0x1.4p-75f);
}
#endif

attribute_hidden float
__math_oflowf (uint32_t sign)
{
  return xflowf (sign, 0x1p97f);
}

attribute_hidden float
__math_divzerof (uint32_t sign)
{
  float y = 0;
  return with_errnof ((sign ? -1 : 1) / y, ERANGE);
}

attribute_hidden float
__math_invalidf (float x)
{
  float y = (x - x) / (x - x);
  return isnan (x) ? y : with_errnof (y, EDOM);
}
