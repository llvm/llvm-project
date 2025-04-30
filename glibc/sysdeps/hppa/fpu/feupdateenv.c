/* Install given floating-point environment and raise exceptions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Huggins-Daines <dhd@debian.org>, 2000

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
#include <string.h>

int
__feupdateenv (const fenv_t *envp)
{
  union { unsigned long long l; unsigned int sw[2]; } s;
  fenv_t temp;
  /* Get the current exception status */
  __asm__ ("fstd %%fr0,0(%1)	\n\t"
           "fldd 0(%1),%%fr0	\n\t"
	   : "=m" (s.l) : "r" (&s.l));

  /* Given environment with exception flags not cleared.  */
  if ((envp != FE_DFL_ENV) && (envp != FE_NOMASK_ENV))
    {
      memcpy(&temp, envp, sizeof (fenv_t));
      temp.__status_word |= s.sw[0] & (FE_ALL_EXCEPT << 27);
    }

  /* Default environment with exception flags not cleared.  */
  if (envp == FE_DFL_ENV)
    temp.__status_word = s.sw[0] & (FE_ALL_EXCEPT << 27);

  /* All traps enabled and current exception flags not cleared.  */
  if (envp == FE_NOMASK_ENV)
    temp.__status_word = (s.sw[0] & (FE_ALL_EXCEPT << 27)) | FE_ALL_EXCEPT;

  /* Install new environment.  */
  __fesetenv (&temp);
  /* Success.  */
  return 0;
}
libm_hidden_def (__feupdateenv)
weak_alias (__feupdateenv, feupdateenv)
libm_hidden_weak (feupdateenv)
