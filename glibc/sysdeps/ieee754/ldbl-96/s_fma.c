/* Compute x * y + z as ternary operation.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2010.

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

#include <float.h>
#include <math.h>
#include <fenv.h>
#include <ieee754.h>
#include <math-barriers.h>
#include <libm-alias-double.h>

/* This implementation uses rounding to odd to avoid problems with
   double rounding.  See a paper by Boldo and Melquiond:
   http://www.lri.fr/~melquion/doc/08-tc.pdf  */

double
__fma (double x, double y, double z)
{
  if (__glibc_unlikely (!isfinite (x) || !isfinite (y)))
    return x * y + z;
  else if (__glibc_unlikely (!isfinite (z)))
    /* If z is Inf, but x and y are finite, the result should be z
       rather than NaN.  */
    return (z + x) + y;

  /* Ensure correct sign of exact 0 + 0.  */
  if (__glibc_unlikely ((x == 0 || y == 0) && z == 0))
    {
      x = math_opt_barrier (x);
      return x * y + z;
    }

  fenv_t env;
  feholdexcept (&env);
  fesetround (FE_TONEAREST);

  /* Multiplication m1 + m2 = x * y using Dekker's algorithm.  */
#define C ((1ULL << (LDBL_MANT_DIG + 1) / 2) + 1)
  long double x1 = (long double) x * C;
  long double y1 = (long double) y * C;
  long double m1 = (long double) x * y;
  x1 = (x - x1) + x1;
  y1 = (y - y1) + y1;
  long double x2 = x - x1;
  long double y2 = y - y1;
  long double m2 = (((x1 * y1 - m1) + x1 * y2) + x2 * y1) + x2 * y2;

  /* Addition a1 + a2 = z + m1 using Knuth's algorithm.  */
  long double a1 = z + m1;
  long double t1 = a1 - z;
  long double t2 = a1 - t1;
  t1 = m1 - t1;
  t2 = z - t2;
  long double a2 = t1 + t2;
  /* Ensure the arithmetic is not scheduled after feclearexcept call.  */
  math_force_eval (m2);
  math_force_eval (a2);
  feclearexcept (FE_INEXACT);

  /* If the result is an exact zero, ensure it has the correct sign.  */
  if (a1 == 0 && m2 == 0)
    {
      feupdateenv (&env);
      /* Ensure that round-to-nearest value of z + m1 is not reused.  */
      z = math_opt_barrier (z);
      return z + m1;
    }

  fesetround (FE_TOWARDZERO);
  /* Perform m2 + a2 addition with round to odd.  */
  a2 = a2 + m2;

  /* Add that to a1 again using rounding to odd.  */
  union ieee854_long_double u;
  u.d = a1 + a2;
  if ((u.ieee.mantissa1 & 1) == 0 && u.ieee.exponent != 0x7fff)
    u.ieee.mantissa1 |= fetestexcept (FE_INEXACT) != 0;
  feupdateenv (&env);

  /* Add finally round to double precision.  */
  return u.d;
}
#ifndef __fma
libm_alias_double (__fma, fma)
#endif
