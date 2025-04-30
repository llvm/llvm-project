/* Raise given exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jes Sorensen <Jes.Sorensen@cern.ch>, 2000.

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
#include <signal.h>
#include <unistd.h>

int
__feraiseexcept (int excepts)
{
  double tmp;
  double dummy;

  /* Raise exceptions represented by EXPECTS.  But we must raise only
     one signal at a time.  It is important the if the overflow/underflow
     exception and the inexact exception are given at the same time,
     the overflow/underflow exception precedes the inexact exception.  */

  /* We do these bits in assembly to be certain GCC doesn't optimize
     away something important.  */

  /* First: invalid exception.  */
  if (FE_INVALID & excepts)
    {
      /* One example of an invalid operation is 0 * Infinity.  */
      tmp = 0;
      __asm__ __volatile__ ("frcpa.s0 %0,p1=f0,f0" : "=f" (tmp) : : "p1" );
    }

  /* Next: division by zero.  */
  if (FE_DIVBYZERO & excepts)
    __asm__ __volatile__ ("frcpa.s0 %0,p1=f1,f0" : "=f" (tmp) : : "p1" );

  /* Next: overflow.  */
  if (FE_OVERFLOW & excepts)
    {
      dummy = DBL_MAX;

      __asm__ __volatile__ ("fadd.d.s0 %0=%1,%1" : "=f" (dummy) : "0" (dummy));
    }

  /* Next: underflow.  */
  if (FE_UNDERFLOW & excepts)
    {
      dummy = DBL_MIN;

      __asm__ __volatile__ ("fnma.d.s0 %0=%1,%1,f0" : "=f" (tmp) : "f" (dummy));
  }

  /* Last: inexact.  */
  if (FE_INEXACT & excepts)
    {
      dummy = DBL_MAX;
      __asm__ __volatile__ ("fsub.d.s0 %0=%1,f1" : "=f" (dummy) : "0" (dummy));
    }

  /* Success.  */
  return 0;
}
libm_hidden_def (__feraiseexcept)
weak_alias (__feraiseexcept, feraiseexcept)
libm_hidden_weak (feraiseexcept)
