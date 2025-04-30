/* Raise given exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Huggins-Daines <dhd@debian.org>

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

/* Please see section 10,
   page 10-5 "Delayed Trapping" in the PA-RISC 2.0 Architecture manual */

int
__feraiseexcept (int excepts)
{
  /* Raise exceptions represented by EXCEPTS.  But we must raise only one
     signal at a time.  It is important that if the overflow/underflow
     exception and the divide by zero exception are given at the same
     time, the overflow/underflow exception follows the divide by zero
     exception.  */

  /* We do these bits in assembly to be certain GCC doesn't optimize
     away something important, and so we can force delayed traps to
     occur. */

  /* We use "fldd 0(%%sr0,%%sp),%0" to flush the delayed exception */

  /* First: Invalid exception.  */
  if (excepts & FE_INVALID)
    {
      /* One example of an invalid operation is 0 * Infinity.  */
      double d = HUGE_VAL;
      __asm__ __volatile__ (
		"	fcpy,dbl %%fr0,%%fr22\n"
		"	fmpy,dbl %0,%%fr22,%0\n"
		"	fldd 0(%%sr0,%%sp),%0"
		: "+f" (d) : : "%fr22" );
    }

  /* Second: Division by zero.  */
  if (excepts & FE_DIVBYZERO)
    {
      double d = 1.0;
      __asm__ __volatile__ (
		"	fcpy,dbl %%fr0,%%fr22\n"
		"	fdiv,dbl %0,%%fr22,%0\n"
		"	fldd 0(%%sr0,%%sp),%0"
		: "+f" (d) : : "%fr22" );
    }

  /* Third: Overflow.  */
  if (excepts & FE_OVERFLOW)
    {
      double d = DBL_MAX;
      __asm__ __volatile__ (
		"	fadd,dbl %0,%0,%0\n"
		"	fldd 0(%%sr0,%%sp),%0"
		: "+f" (d) );
    }

  /* Fourth: Underflow.  */
  if (excepts & FE_UNDERFLOW)
    {
      double d = DBL_MIN;
      double e = 3.0;
      __asm__ __volatile__ (
		"	fdiv,dbl %0,%1,%0\n"
		"	fldd 0(%%sr0,%%sp),%0"
		: "+f" (d) : "f" (e) );
    }

  /* Fifth: Inexact */
  if (excepts & FE_INEXACT)
    {
      double d = M_PI;
      double e = 69.69;
      __asm__ __volatile__ (
		"	fdiv,dbl %0,%1,%%fr22\n"
		"	fcnvfxt,dbl,sgl %%fr22,%%fr22L\n"
		"	fldd 0(%%sr0,%%sp),%%fr22"
		: : "f" (d), "f" (e) : "%fr22" );
    }

  /* Success.  */
  return 0;
}
libm_hidden_def (__feraiseexcept)
weak_alias (__feraiseexcept, feraiseexcept)
libm_hidden_weak (feraiseexcept)
