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

#include <math.h>
#include <fenv.h>
#include <ieee754.h>
#include <libm-alias-double.h>
#include <math-use-builtins.h>

/* This implementation relies on long double being more than twice as
   precise as double and uses rounding to odd in order to avoid problems
   with double rounding.
   See a paper by Boldo and Melquiond:
   http://www.lri.fr/~melquion/doc/08-tc.pdf  */

double
__fma (double x, double y, double z)
{
#if USE_FMA_BUILTIN
  return __builtin_fma (x, y, z);
#else
  fenv_t env;
  /* Multiplication is always exact.  */
  long double temp = (long double) x * (long double) y;

  /* Ensure correct sign of an exact zero result by performing the
     addition in the original rounding mode in that case.  */
  if (temp == -z)
    return (double) temp + z;

  union ieee854_long_double u;
  feholdexcept (&env);
  fesetround (FE_TOWARDZERO);
  /* Perform addition with round to odd.  */
  u.d = temp + (long double) z;
  if ((u.ieee.mantissa3 & 1) == 0 && u.ieee.exponent != 0x7fff)
    u.ieee.mantissa3 |= fetestexcept (FE_INEXACT) != 0;
  feupdateenv (&env);
  /* And finally truncation with round to nearest.  */
  return (double) u.d;
#endif /* ! USE_FMA_BUILTIN  */
}
#ifndef __fma
libm_alias_double (__fma, fma)
#endif
