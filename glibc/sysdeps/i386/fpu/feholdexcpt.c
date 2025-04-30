/* Store current floating-point environment and clear exceptions.
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
__feholdexcept (fenv_t *envp)
{
  /* Store the environment.  Recall that fnstenv has a side effect of
     masking all exceptions.  Then clear all exceptions.  */
  __asm__ volatile ("fnstenv %0; fnclex" : "=m" (*envp));

  /* If the CPU supports SSE we set the MXCSR as well.  */
  if (CPU_FEATURE_USABLE (SSE))
    {
      unsigned int xwork;

      /* Get the current control word.  */
      __asm__ ("stmxcsr %0" : "=m" (envp->__eip));

      /* Set all exceptions to non-stop and clear them.  */
      xwork = (envp->__eip | 0x1f80) & ~0x3f;

      __asm__ ("ldmxcsr %0" : : "m" (*&xwork));
    }

  return 0;
}
libm_hidden_def (__feholdexcept)
weak_alias (__feholdexcept, feholdexcept)
libm_hidden_weak (feholdexcept)
