/* Install given floating-point environment.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <fenv_private.h>
#include <shlib-compat.h>

int
__fesetenv (const fenv_t *envp)
{
  fenv_t dummy;

  /* Put these constants in memory explicitly, so as to cope with a
     -fPIC bug as of gcc 970624.  Making them automatic is quicker
     than loading up the pic register in this instance.  */

  if (envp == FE_DFL_ENV)
    {
      dummy = 0;
      envp = &dummy;
    }
  else if (envp == FE_NOMASK_ENV)
    {
      dummy = 0x1f << 23;
      envp = &dummy;
    }

  __fenv_ldfsr (*envp);

  /* Success.  */
  return 0;
}

#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__fesetenv, __old_fesetenv)
compat_symbol (libm, __old_fesetenv, fesetenv, GLIBC_2_1);
#endif

libm_hidden_def (__fesetenv)
libm_hidden_ver (__fesetenv, fesetenv)
versioned_symbol (libm, __fesetenv, fesetenv, GLIBC_2_2);
