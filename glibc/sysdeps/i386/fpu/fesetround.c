/* Set current rounding direction.
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
#include <unistd.h>
#include <ldsodefs.h>
#include <dl-procinfo.h>

int
__fesetround (int round)
{
  unsigned short int cw;

  if ((round & ~0xc00) != 0)
    /* ROUND is no valid rounding mode.  */
    return 1;

  __asm__ ("fnstcw %0" : "=m" (*&cw));
  cw &= ~0xc00;
  cw |= round;
  __asm__ ("fldcw %0" : : "m" (*&cw));

  /* If the CPU supports SSE we set the MXCSR as well.  */
  if (CPU_FEATURE_USABLE (SSE))
    {
      unsigned int xcw;

      __asm__ ("stmxcsr %0" : "=m" (*&xcw));
      xcw &= ~0x6000;
      xcw |= round << 3;
      __asm__ ("ldmxcsr %0" : : "m" (*&xcw));
    }

  return 0;
}
libm_hidden_def (__fesetround)
weak_alias (__fesetround, fesetround)
libm_hidden_weak (fesetround)
