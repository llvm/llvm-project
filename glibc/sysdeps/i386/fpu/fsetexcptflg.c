/* Set floating-point environment exception handling.
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
#include <unistd.h>
#include <ldsodefs.h>
#include <dl-procinfo.h>

int
__fesetexceptflag (const fexcept_t *flagp, int excepts)
{
  fenv_t temp;

  /* Get the current environment.  We have to do this since we cannot
     separately set the status word.  */
  __asm__ ("fnstenv %0" : "=m" (*&temp));

  temp.__status_word &= ~(excepts & FE_ALL_EXCEPT);
  temp.__status_word |= *flagp & excepts & FE_ALL_EXCEPT;

  /* Store the new status word (along with the rest of the environment.
     Possibly new exceptions are set but they won't get executed unless
     the next floating-point instruction.  */
  __asm__ ("fldenv %0" : : "m" (*&temp));

  /* If the CPU supports SSE, we set the MXCSR as well.  */
  if (CPU_FEATURE_USABLE (SSE))
    {
      unsigned int xnew_exc;

      /* Get the current MXCSR.  */
      __asm__ ("stmxcsr %0" : "=m" (*&xnew_exc));

      /* Set the relevant bits.  */
      xnew_exc &= ~(excepts & FE_ALL_EXCEPT);
      xnew_exc |= *flagp & excepts & FE_ALL_EXCEPT;

      /* Put the new data in effect.  */
      __asm__ ("ldmxcsr %0" : : "m" (*&xnew_exc));
    }

  /* Success.  */
  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__fesetexceptflag, __old_fesetexceptflag)
compat_symbol (libm, __old_fesetexceptflag, fesetexceptflag, GLIBC_2_1);
#endif

versioned_symbol (libm, __fesetexceptflag, fesetexceptflag, GLIBC_2_2);
