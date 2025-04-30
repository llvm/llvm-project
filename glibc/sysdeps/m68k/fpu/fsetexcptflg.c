/* Set floating-point environment exception handling.
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
#include <math.h>

int
__fesetexceptflag (const fexcept_t *flagp, int excepts)
{
  fexcept_t fpsr;

  /* Get the current status register.  */
  __asm__ ("fmove%.l %/fpsr,%0" : "=dm" (fpsr));

  /* Install the new exception bits in the Accrued Exception Byte.  */
  fpsr &= ~(excepts & FE_ALL_EXCEPT);
  fpsr |= *flagp & excepts & FE_ALL_EXCEPT;

  /* Store the new status register.  */
  __asm__ __volatile__ ("fmove%.l %0,%/fpsr" : : "dm" (fpsr));

  /* Success.  */
  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__fesetexceptflag, __old_fesetexceptflag)
compat_symbol (libm, __old_fesetexceptflag, fesetexceptflag, GLIBC_2_1);
#endif

versioned_symbol (libm, __fesetexceptflag, fesetexceptflag, GLIBC_2_2);
