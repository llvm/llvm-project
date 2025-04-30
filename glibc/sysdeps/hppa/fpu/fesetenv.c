/* Install given floating-point environment.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Huggins-Daines <dhd@debian.org>, 2000
   Based on the m68k version by
   Andreas Schwab <schwab@suse.de>

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
__fesetenv (const fenv_t *envp)
{
  union { unsigned long long buf[4]; fenv_t env; } temp;
  unsigned long long *bufptr;

  /* Install the environment specified by ENVP.  But there are a few
     values which we do not want to come from the saved environment.
     Therefore, we get the current environment and replace the values
     we want to use from the environment specified by the parameter.  */
  bufptr = temp.buf;
  __asm__ (
	   "fstd %%fr0,0(%1)\n"
	   : "=m" (temp) : "r" (bufptr) : "%r0");

  temp.env.__status_word &= ~(FE_ALL_EXCEPT
			    | (FE_ALL_EXCEPT << 27)
			    | FE_DOWNWARD);
  if (envp == FE_DFL_ENV)
    temp.env.__status_word = 0;
  else if (envp == FE_NOMASK_ENV)
    temp.env.__status_word |= FE_ALL_EXCEPT;
  else
    temp.env.__status_word |= (envp->__status_word
			       & (FE_ALL_EXCEPT
				  | FE_DOWNWARD
				  | (FE_ALL_EXCEPT << 27)));

  /* Load the new environment. We use bufptr again since the
     initial asm has modified the value of the register and here
     we take advantage of that to load in reverse order so fr0
     is loaded last and T-Bit is enabled. */
  __asm__ (
	   "fldd 0(%1),%%fr0\n"
	   : : "m" (temp), "r" (bufptr) : "%r0" );

  /* Success.  */
  return 0;
}
libm_hidden_def (__fesetenv)
weak_alias (__fesetenv, fesetenv)
libm_hidden_weak (fesetenv)
