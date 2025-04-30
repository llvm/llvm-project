/* Store current floating-point environment and clear exceptions.
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

int
__feholdexcept (fenv_t *envp)
{
  fexcept_t fpcr, fpsr;

  /* Store the environment.  */
#ifdef __mcoldfire__
  __asm__ ("fmove%.l %/fpcr,%0" : "=dm" (envp->__control_register));
  __asm__ ("fmove%.l %/fpsr,%0" : "=dm" (envp->__status_register));
  __asm__ ("fmove%.l %/fpiar,%0" : "=dm" (envp->__instruction_address));
#else
  __asm__ ("fmovem%.l %/fpcr/%/fpsr/%/fpiar,%0" : "=m" (*envp));
#endif

  /* Now clear all exceptions.  */
  fpsr = envp->__status_register & ~FE_ALL_EXCEPT;
  __asm__ __volatile__ ("fmove%.l %0,%/fpsr" : : "dm" (fpsr));
  /* And set all exceptions to non-stop.  */
  fpcr = envp->__control_register & ~(FE_ALL_EXCEPT << 6);
  __asm__ __volatile__ ("fmove%.l %0,%!" : : "dm" (fpcr));

  return 0;
}
libm_hidden_def (__feholdexcept)
weak_alias (__feholdexcept, feholdexcept)
libm_hidden_weak (feholdexcept)
