/* Test exception in current environment.
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
#include <dl-procinfo.h>
#include <ldsodefs.h>

int
fetestexcept (int excepts)
{
  short temp;
  int xtemp = 0;

  /* Get current exceptions.  */
  __asm__ ("fnstsw %0" : "=a" (temp));

  /* If the CPU supports SSE we test the MXCSR as well.  */
  if (CPU_FEATURE_USABLE (SSE))
    __asm__ ("stmxcsr %0" : "=m" (*&xtemp));

  return (temp | xtemp) & excepts & FE_ALL_EXCEPT;
}
libm_hidden_def (fetestexcept)
