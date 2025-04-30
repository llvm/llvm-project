/* Set floating-point environment exception handling.
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
fesetexceptflag (const fexcept_t *flagp, int excepts)
{
  fenv_t temp;
  unsigned int mxcsr;

  /* XXX: Do we really need to set both the exception in both units?
     Shouldn't it be enough to set only the SSE unit?  */

  /* Get the current x87 FPU environment.  We have to do this since we
     cannot separately set the status word.  */
  __asm__ ("fnstenv %0" : "=m" (*&temp));

  temp.__status_word &= ~(excepts & FE_ALL_EXCEPT);
  temp.__status_word |= *flagp & excepts & FE_ALL_EXCEPT;

  /* Store the new status word (along with the rest of the environment.
     Possibly new exceptions are set but they won't get executed unless
     the next floating-point instruction.  */
  __asm__ ("fldenv %0" : : "m" (*&temp));

  /* And now the same for SSE.  */
  __asm__ ("stmxcsr %0" : "=m" (*&mxcsr));

  mxcsr &= ~(excepts & FE_ALL_EXCEPT);
  mxcsr |= *flagp & excepts & FE_ALL_EXCEPT;

  __asm__ ("ldmxcsr %0" : : "m" (*&mxcsr));

  /* Success.  */
  return 0;
}
