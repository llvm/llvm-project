/* Install given floating-point environment and raise exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Christian Boissat <Christian.Boissat@cern.ch>, 1999.

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

int
__feupdateenv (const fenv_t *envp)
{
  fenv_t fpsr;


  /* Get the current exception state.  */
  __asm__ __volatile__ ("mov.m %0=ar.fpsr" : "=r" (fpsr));

  /* Install new environment.  */
  __fesetenv (envp);

  /* Raise the saved exceptions.  */
  __feraiseexcept ((int) (fpsr >> 13) & FE_ALL_EXCEPT);

  /* Success.  */
  return 0;
}
libm_hidden_def (__feupdateenv)
weak_alias (__feupdateenv, feupdateenv)
libm_hidden_weak (feupdateenv)
