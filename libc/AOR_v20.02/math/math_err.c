/*
 * Double-precision math error handling.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

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
  y = eval_as_double (opt_barrier_double (sign ? -y : y) * y);
  return with_errno (y, ERANGE);
}

HIDDEN double
__math_uflow (uint32_t sign)
{
  return xflow (sign, 0x1p-767);
}

#if WANT_ERRNO_UFLOW
/* Underflows to zero in some non-nearest rounding mode, setting errno
   is valid even if the result is non-zero, but in the subnormal range.  */
HIDDEN double
__math_may_uflow (uint32_t sign)
{
  return xflow (sign, 0x1.8p-538);
}
#endif

HIDDEN double
__math_oflow (uint32_t sign)
{
  return xflow (sign, 0x1p769);
}

HIDDEN double
__math_divzero (uint32_t sign)
{
  double y = opt_barrier_double (sign ? -1.0 : 1.0) / 0.0;
  return with_errno (y, ERANGE);
}

HIDDEN double
__math_invalid (double x)
{
  double y = (x - x) / (x - x);
  return isnan (x) ? y : with_errno (y, EDOM);
}

/* Check result and set errno if necessary.  */

HIDDEN double
__math_check_uflow (double y)
{
  return y == 0.0 ? with_errno (y, ERANGE) : y;
}

HIDDEN double
__math_check_oflow (double y)
{
  return isinf (y) ? with_errno (y, ERANGE) : y;
}
