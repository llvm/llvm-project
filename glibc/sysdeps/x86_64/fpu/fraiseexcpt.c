/* Raise given exceptions.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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
#include <math.h>

int
__feraiseexcept (int excepts)
{
  /* Raise exceptions represented by EXPECTS.  But we must raise only
     one signal at a time.  It is important that if the overflow/underflow
     exception and the inexact exception are given at the same time,
     the overflow/underflow exception follows the inexact exception.  */

  /* First: invalid exception.  */
  if ((FE_INVALID & excepts) != 0)
    {
      /* One example of an invalid operation is 0.0 / 0.0.  */
      float f = 0.0;

      __asm__ __volatile__ ("divss %0, %0 " : : "x" (f));
      (void) &f;
    }

  /* Next: division by zero.  */
  if ((FE_DIVBYZERO & excepts) != 0)
    {
      float f = 1.0;
      float g = 0.0;

      __asm__ __volatile__ ("divss %1, %0" : : "x" (f), "x" (g));
      (void) &f;
    }

  /* Next: overflow.  */
  if ((FE_OVERFLOW & excepts) != 0)
    {
      /* XXX: Is it ok to only set the x87 FPU?  */
      /* There is no way to raise only the overflow flag.  Do it the
	 hard way.  */
      fenv_t temp;

      /* Bah, we have to clear selected exceptions.  Since there is no
	 `fldsw' instruction we have to do it the hard way.  */
      __asm__ __volatile__ ("fnstenv %0" : "=m" (*&temp));

      /* Set the relevant bits.  */
      temp.__status_word |= FE_OVERFLOW;

      /* Put the new data in effect.  */
      __asm__ __volatile__ ("fldenv %0" : : "m" (*&temp));

      /* And raise the exception.  */
      __asm__ __volatile__ ("fwait");
    }

  /* Next: underflow.  */
  if ((FE_UNDERFLOW & excepts) != 0)
    {
      /* XXX: Is it ok to only set the x87 FPU?  */
      /* There is no way to raise only the underflow flag.  Do it the
	 hard way.  */
      fenv_t temp;

      /* Bah, we have to clear selected exceptions.  Since there is no
	 `fldsw' instruction we have to do it the hard way.  */
      __asm__ __volatile__ ("fnstenv %0" : "=m" (*&temp));

      /* Set the relevant bits.  */
      temp.__status_word |= FE_UNDERFLOW;

      /* Put the new data in effect.  */
      __asm__ __volatile__ ("fldenv %0" : : "m" (*&temp));

      /* And raise the exception.  */
      __asm__ __volatile__ ("fwait");
    }

  /* Last: inexact.  */
  if ((FE_INEXACT & excepts) != 0)
    {
      /* XXX: Is it ok to only set the x87 FPU?  */
      /* There is no way to raise only the inexact flag.  Do it the
	 hard way.  */
      fenv_t temp;

      /* Bah, we have to clear selected exceptions.  Since there is no
	 `fldsw' instruction we have to do it the hard way.  */
      __asm__ __volatile__ ("fnstenv %0" : "=m" (*&temp));

      /* Set the relevant bits.  */
      temp.__status_word |= FE_INEXACT;

      /* Put the new data in effect.  */
      __asm__ __volatile__ ("fldenv %0" : : "m" (*&temp));

      /* And raise the exception.  */
      __asm__ __volatile__ ("fwait");
    }

  /* Success.  */
  return 0;
}
libm_hidden_def (__feraiseexcept)
weak_alias (__feraiseexcept, feraiseexcept)
libm_hidden_weak (feraiseexcept)
