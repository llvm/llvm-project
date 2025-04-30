/* Raise given exceptions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Denis Joseph Barrow (djbarrow@de.ibm.com) and
   Martin Schwidefsky (schwidefsky@de.ibm.com).

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

#include <fenv_libc.h>
#include <float.h>
#include <math.h>


static __inline__ void
fexceptdiv (float d, float e)
{
  __asm__ __volatile__ ("debr %0,%1" : : "f" (d), "f" (e) );
}

static __inline__ void
fexceptadd (float d, float e)
{
  __asm__ __volatile__ ("aebr %0,%1" : : "f" (d), "f" (e) );
}

#ifdef HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
static __inline__ void
fexceptround (double e)
{
  float d;
  /* Load rounded from double to float with M3 = round toward 0, M4 = Suppress
     IEEE-inexact exception.
     In case of e=0x1p128 and the overflow-mask bit is zero, only the
     IEEE-overflow flag is set. If overflow-mask bit is one, DXC field is set to
     0x20 "IEEE overflow, exact".
     In case of e=0x1p-150 and the underflow-mask bit is zero, only the
     IEEE-underflow flag is set. If underflow-mask bit is one, DXC field is set
     to 0x10 "IEEE underflow, exact".
     This instruction is available with a zarch machine >= z196.  */
  __asm__ __volatile__ ("ledbra %0,5,%1,4" : "=f" (d) : "f" (e) );
}
#endif

int
__feraiseexcept (int excepts)
{
  /* Raise exceptions represented by EXPECTS.  But we must raise only
     one signal at a time.  It is important that if the overflow/underflow
     exception and the inexact exception are given at the same time,
     the overflow/underflow exception follows the inexact exception.  */

  /* First: invalid exception.  */
  if (FE_INVALID & excepts)
    fexceptdiv (0.0, 0.0);

  /* Next: division by zero.  */
  if (FE_DIVBYZERO & excepts)
    fexceptdiv (1.0, 0.0);

  /* Next: overflow.  */
  if (FE_OVERFLOW & excepts)
    {
#ifdef HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
      fexceptround (0x1p128);
#else
      /* If overflow-mask bit is zero, both IEEE-overflow and IEEE-inexact flags
	 are set.  If overflow-mask bit is one, DXC field is set to 0x2C "IEEE
	 overflow, inexact and incremented".  */
      fexceptadd (FLT_MAX, 1.0e32);
#endif
    }

  /* Next: underflow.  */
  if (FE_UNDERFLOW & excepts)
    {
#ifdef HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
      fexceptround (0x1p-150);
#else
      /* If underflow-mask bit is zero, both IEEE-underflow and IEEE-inexact
	 flags are set.  If underflow-mask bit is one, DXC field is set to 0x1C
	 "IEEE underflow, inexact and incremented".  */
      fexceptdiv (FLT_MIN, 3.0);
#endif
    }

  /* Last: inexact.  */
  if (FE_INEXACT & excepts)
    fexceptdiv (2.0, 3.0);

  /* Success.  */
  return 0;
}
libm_hidden_def (__feraiseexcept)
weak_alias (__feraiseexcept, feraiseexcept)
libm_hidden_weak (feraiseexcept)
