/* Raise given exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
      double d;
      __asm__ __volatile__ ("fldz; fdiv %%st, %%st(0); fwait" : "=t" (d));
      (void) &d;
    }

  /* Next: division by zero.  */
  if ((FE_DIVBYZERO & excepts) != 0)
    {
      double d;
      __asm__ __volatile__ ("fldz; fld1; fdivp %%st, %%st(1); fwait"
			    : "=t" (d));
      (void) &d;
    }

  /* Next: overflow.  */
  if ((FE_OVERFLOW & excepts) != 0)
    {
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

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__feraiseexcept, __old_feraiseexcept)
compat_symbol (libm, __old_feraiseexcept, feraiseexcept, GLIBC_2_1);
#endif

libm_hidden_def (__feraiseexcept)
libm_hidden_ver (__feraiseexcept, feraiseexcept)
versioned_symbol (libm, __feraiseexcept, feraiseexcept, GLIBC_2_2);
