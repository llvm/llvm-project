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
#include <math-barriers.h>
#include <fenv_private.h>
#include <libm-alias-float.h>
#include <math-use-builtins.h>

/* This implementation relies on double being more than twice as
   precise as float and uses rounding to odd in order to avoid problems
   with double rounding.
   See a paper by Boldo and Melquiond:
   http://www.lri.fr/~melquion/doc/08-tc.pdf  */

float
__fmaf (float x, float y, float z)
{
#if USE_FMAF_BUILTIN
  return __builtin_fmaf (x, y, z);
#else
  /* Use generic implementation.  */
  fenv_t env;

  /* Multiplication is always exact.  */
  double temp = (double) x * (double) y;

  /* Ensure correct sign of an exact zero result by performing the
     addition in the original rounding mode in that case.  */
  if (temp == -z)
    return (float) temp + z;

  union ieee754_double u;

  libc_feholdexcept_setround (&env, FE_TOWARDZERO);

  /* Perform addition with round to odd.  */
  u.d = temp + (double) z;
  /* Ensure the addition is not scheduled after fetestexcept call.  */
  math_force_eval (u.d);

  /* Reset rounding mode and test for inexact simultaneously.  */
  int j = libc_feupdateenv_test (&env, FE_INEXACT) != 0;

  if ((u.ieee.mantissa1 & 1) == 0 && u.ieee.exponent != 0x7ff)
    u.ieee.mantissa1 |= j;

  /* And finally truncation with round to nearest.  */
  return (float) u.d;
#endif /* ! USE_FMAF_BUILTIN  */
}
#ifndef __fmaf
libm_alias_float (__fma, fma)
#endif
