/* Raise given exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <shlib-compat.h>

int
__feraiseexcept (int excepts)
{
  static const struct {
    double zero, one, max, min, pi;
  } c = {
    0.0, 1.0, DBL_MAX, DBL_MIN, M_PI
  };
  double d;

  /* Raise exceptions represented by EXPECTS.  But we must raise only
     one signal at a time.  It is important the if the overflow/underflow
     exception and the inexact exception are given at the same time,
     the overflow/underflow exception follows the inexact exception.  */

  /* First: invalid exception.  */
  if ((FE_INVALID & excepts) != 0)
    {
      /* One example of an invalid operation is 0/0.  */
      __asm ("" : "=e" (d) : "0" (c.zero));
      d /= c.zero;
      __asm __volatile ("" : : "e" (d));
    }

  /* Next: division by zero.  */
  if ((FE_DIVBYZERO & excepts) != 0)
    {
      __asm ("" : "=e" (d) : "0" (c.one));
      d /= c.zero;
      __asm __volatile ("" : : "e" (d));
    }

  /* Next: overflow.  */
  if ((FE_OVERFLOW & excepts) != 0)
    {
      __asm ("" : "=e" (d) : "0" (c.max));
      d *= d;
      __asm __volatile ("" : : "e" (d));
    }

  /* Next: underflow.  */
  if ((FE_UNDERFLOW & excepts) != 0)
    {
      __asm ("" : "=e" (d) : "0" (c.min));
      d *= d;
      __asm __volatile ("" : : "e" (d));
    }

  /* Last: inexact.  */
  if ((FE_INEXACT & excepts) != 0)
    {
      __asm ("" : "=e" (d) : "0" (c.one));
      d /= c.pi;
      __asm __volatile ("" : : "e" (d));
    }

  /* Success.  */
  return 0;
}

#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__feraiseexcept, __old_feraiseexcept)
compat_symbol (libm, __old_feraiseexcept, feraiseexcept, GLIBC_2_1);
#endif

libm_hidden_def (__feraiseexcept)
libm_hidden_ver (__feraiseexcept, feraiseexcept)
versioned_symbol (libm, __feraiseexcept, feraiseexcept, GLIBC_2_2);
