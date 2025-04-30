/* Set the FPU control word for x86.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <fpu_control.h>
#include <fenv.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <dl-procinfo.h>

void
__setfpucw (fpu_control_t set)
{
  fpu_control_t cw;

  /* Fetch the current control word.  */
  __asm__ ("fnstcw %0" : "=m" (*&cw));

  /* Preserve the reserved bits, and set the rest as the user
     specified (or the default, if the user gave zero).  */
  cw &= _FPU_RESERVED;
  cw |= set & ~_FPU_RESERVED;

  __asm__ ("fldcw %0" : : "m" (*&cw));

  /* If the CPU supports SSE, we set the MXCSR as well.  */
  if (CPU_FEATURE_USABLE (SSE))
    {
      unsigned int xnew_exc;

      /* Get the current MXCSR.  */
      __asm__ ("stmxcsr %0" : "=m" (*&xnew_exc));

      xnew_exc &= ~((0xc00 << 3) | (FE_ALL_EXCEPT << 7));
      xnew_exc |= ((set & 0xc00) << 3) | ((set & FE_ALL_EXCEPT) << 7);

      __asm__ ("ldmxcsr %0" : : "m" (*&xnew_exc));
    }
}
