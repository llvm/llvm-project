/* Install given floating-point environment and raise exceptions.
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

#include <fenv_libc.h>
#include <fpu_control.h>

int
__feupdateenv (const fenv_t *envp)
{
  fenv_union_t old, new;

  /* Save the currently set exceptions.  */
  new.fenv = *envp;
  old.fenv = fegetenv_register ();

  /* Restore rounding mode and exception enable from *envp and merge
     exceptions.  Leave fraction rounded/inexact and FP result/CC bits
     unchanged.  */
  new.l = (old.l & 0xffffffff1fffff00LL) | (new.l & 0x1ff80fff);

  __TEST_AND_EXIT_NON_STOP (old.l, new.l);
  __TEST_AND_ENTER_NON_STOP (old.l, new.l);

  /* Atomically enable and raise (if appropriate) exceptions set in `new'. */
  fesetenv_register (new.fenv);

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
