/*
 * Single-precision math error handling.
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
NOINLINE static float
with_errnof (float y, int e)
{
  errno = e;
  return y;
}
#else
#define with_errnof(x, e) (x)
#endif

/* NOINLINE reduces code size.  */
NOINLINE static float
xflowf (uint32_t sign, float y)
{
  y = eval_as_float (opt_barrier_float (sign ? -y : y) * y);
  return with_errnof (y, ERANGE);
}

HIDDEN float
__math_uflowf (uint32_t sign)
{
  return xflowf (sign, 0x1p-95f);
}

#if WANT_ERRNO_UFLOW
/* Underflows to zero in some non-nearest rounding mode, setting errno
   is valid even if the result is non-zero, but in the subnormal range.  */
HIDDEN float
__math_may_uflowf (uint32_t sign)
{
  return xflowf (sign, 0x1.4p-75f);
}
#endif

HIDDEN float
__math_oflowf (uint32_t sign)
{
  return xflowf (sign, 0x1p97f);
}

HIDDEN float
__math_divzerof (uint32_t sign)
{
  float y = opt_barrier_float (sign ? -1.0f : 1.0f) / 0.0f;
  return with_errnof (y, ERANGE);
}

HIDDEN float
__math_invalidf (float x)
{
  float y = (x - x) / (x - x);
  return isnan (x) ? y : with_errnof (y, EDOM);
}
