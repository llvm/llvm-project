/* Double-precision math error handling.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <math-barriers.h>
#include "math_config.h"

#if WANT_ERRNO
#include <errno.h>
/* NOINLINE reduces code size and avoids making math functions non-leaf
   when the error handling is inlined.  */
NOINLINE static double
with_errno (double y, int e)
{
  errno = e;
  return y;
}
#else
#define with_errno(x, e) (x)
#endif

/* NOINLINE reduces code size.  */
NOINLINE static double
xflow (uint32_t sign, double y)
{
  y = math_opt_barrier (sign ? -y : y) * y;
  return with_errno (y, ERANGE);
}

attribute_hidden double
__math_uflow (uint32_t sign)
{
  return xflow (sign, 0x1p-767);
}

#if WANT_ERRNO_UFLOW
/* Underflows to zero in some non-nearest rounding mode, setting errno
   is valid even if the result is non-zero, but in the subnormal range.  */
attribute_hidden double
__math_may_uflow (uint32_t sign)
{
  return xflow (sign, 0x1.8p-538);
}
#endif

attribute_hidden double
__math_oflow (uint32_t sign)
{
  return xflow (sign, 0x1p769);
}

attribute_hidden double
__math_divzero (uint32_t sign)
{
  double y = math_opt_barrier (sign ? -1.0 : 1.0) / 0.0;
  return with_errno (y, ERANGE);
}

attribute_hidden double
__math_invalid (double x)
{
  double y = (x - x) / (x - x);
  return isnan (x) ? y : with_errno (y, EDOM);
}

/* Check result and set errno if necessary.  */

attribute_hidden double
__math_check_uflow (double y)
{
  return y == 0.0 ? with_errno (y, ERANGE) : y;
}

attribute_hidden double
__math_check_oflow (double y)
{
  return isinf (y) ? with_errno (y, ERANGE) : y;
}
