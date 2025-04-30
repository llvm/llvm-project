/* Install given floating-point environment and raise exceptions.
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
__feupdateenv (const fenv_t *envp)
{
  fexcept_t fpsr;

  /* Save current exceptions.  */
  __asm__ ("fmove%.l %/fpsr,%0" : "=dm" (fpsr));
  fpsr &= FE_ALL_EXCEPT;

  /* Install new environment.  */
  __fesetenv (envp);

  /* Raise the saved exception.  Incidently for us the implementation
     defined format of the values in objects of type fexcept_t is the
     same as the ones specified using the FE_* constants.  */
  __feraiseexcept ((int) fpsr);

  /* Success.  */
  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__feupdateenv, __old_feupdateenv)
compat_symbol (libm, __old_feupdateenv, feupdateenv, GLIBC_2_1);
#endif

libm_hidden_def (__feupdateenv)
libm_hidden_ver (__feupdateenv, feupdateenv)
versioned_symbol (libm, __feupdateenv, feupdateenv, GLIBC_2_2);
