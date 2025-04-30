/* Compute sine and cosine of argument.
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

#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <math-barriers.h>
#include <libm-alias-float.h>
#include "math_config.h"
#include "s_sincosf.h"

#ifndef SINCOSF
# define SINCOSF_FUNC __sincosf
#else
# define SINCOSF_FUNC SINCOSF
#endif

/* Fast sincosf implementation.  Worst-case ULP is 0.5607, maximum relative
   error is 0.5303 * 2^-23.  A single-step range reduction is used for
   small values.  Large inputs have their range reduced using fast integer
   arithmetic.  */
void
SINCOSF_FUNC (float y, float *sinp, float *cosp)
{
  double x = y;
  double s;
  int n;
  const sincos_t *p = &__sincosf_table[0];

  if (abstop12 (y) < abstop12 (pio4))
    {
      double x2 = x * x;

      if (__glibc_unlikely (abstop12 (y) < abstop12 (0x1p-12f)))
      {
	/* Force underflow for tiny y.  */
	if (__glibc_unlikely (abstop12 (y) < abstop12 (0x1p-126f)))
	  math_force_eval ((float)x2);
	*sinp = y;
	*cosp = 1.0f;
	return;
      }

      sincosf_poly (x, x2, p, 0, sinp, cosp);
    }
  else if (abstop12 (y) < abstop12 (120.0f))
    {
      x = reduce_fast (x, p, &n);

      /* Setup the signs for sin and cos.  */
      s = p->sign[n & 3];

      if (n & 2)
	p = &__sincosf_table[1];

      sincosf_poly (x * s, x * x, p, n, sinp, cosp);
    }
  else if (__glibc_likely (abstop12 (y) < abstop12 (INFINITY)))
    {
      uint32_t xi = asuint (y);
      int sign = xi >> 31;

      x = reduce_large (xi, &n);

      /* Setup signs for sin and cos - include original sign.  */
      s = p->sign[(n + sign) & 3];

      if ((n + sign) & 2)
	p = &__sincosf_table[1];

      sincosf_poly (x * s, x * x, p, n, sinp, cosp);
    }
  else
    {
      /* Return NaN if Inf or NaN for both sin and cos.  */
      *sinp = *cosp = y - y;
#if WANT_ERRNO
      /* Needed to set errno for +-Inf, the add is a hack to work
	 around a gcc register allocation issue: just passing y
	 affects code generation in the fast path (PR86901).  */
      __math_invalidf (y + y);
#endif
    }
}

#ifndef SINCOSF
libm_alias_float (__sincos, sincos)
#endif
