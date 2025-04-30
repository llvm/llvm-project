/* Raise given exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@issan.informatik.uni-dortmund.de>

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv.h>
#include <float.h>
#include <math.h>

int
__feraiseexcept (int excepts)
{
  /* Raise exceptions represented by EXCEPTS.  But we must raise only one
     signal at a time.  It is important that if the overflow/underflow
     exception and the divide by zero exception are given at the same
     time, the overflow/underflow exception follows the divide by zero
     exception.  */

  /* First: invalid exception.  */
  if (excepts & FE_INVALID)
    {
      /* One example of an invalid operation is 0 * Infinity.  */
      double d = HUGE_VAL;
      __asm__ __volatile__ ("fmul%.s %#0r0,%0; fnop" : "=f" (d) : "0" (d));
    }

  /* Next: division by zero.  */
  if (excepts & FE_DIVBYZERO)
    {
      double d = 1.0;
      __asm__ __volatile__ ("fdiv%.s %#0r0,%0; fnop" : "=f" (d) : "0" (d));
    }

  /* Next: overflow.  */
  if (excepts & FE_OVERFLOW)
    {
      long double d = LDBL_MAX;

      __asm__ __volatile__ ("fmul%.x %0,%0; fnop" : "=f" (d) : "0" (d));
    }

  /* Next: underflow.  */
  if (excepts & FE_UNDERFLOW)
    {
      long double d = -LDBL_MAX;

      __asm__ __volatile__ ("fetox%.x %0; fnop" : "=f" (d) : "0" (d));
    }

  /* Last: inexact.  */
  if (excepts & FE_INEXACT)
    {
      long double d = 1.0;
      __asm__ __volatile__ ("fdiv%.s %#0r3,%0; fnop" : "=f" (d) : "0" (d));
    }

  /* Success.  */
  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__feraiseexcept, __old_feraiseexcept)
compat_symbol (libm, __old_feraiseexcept, feraiseexcept, GLIBC_2_1);
#endif

libm_hidden_def (__feraiseexcept)
libm_hidden_ver (__feraiseexcept, feraiseexcept)
versioned_symbol (libm, __feraiseexcept, feraiseexcept, GLIBC_2_2);
