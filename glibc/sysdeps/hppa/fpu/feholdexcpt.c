/* Store current floating-point environment and clear exceptions.
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
__feholdexcept (fenv_t *envp)
{
  union { unsigned long long buf[4]; fenv_t env; } clear;
  unsigned long long *bufptr;

  /* Store the environment.  */
  bufptr = clear.buf;
  __asm__ (
	   "fstd %%fr0,0(%1)\n"
	   : "=m" (clear) : "r" (bufptr) : "%r0");
  memcpy (envp, &clear.env, sizeof (fenv_t));

  /* Clear exception queues */
  memset (clear.env.__exception, 0, sizeof (clear.env.__exception));
  /* And set all exceptions to non-stop.  */
  clear.env.__status_word &= ~FE_ALL_EXCEPT;
  /* Now clear all flags  */
  clear.env.__status_word &= ~(FE_ALL_EXCEPT << 27);

  /* Load the new environment. Note: fr0 must load last to enable T-bit.  */
  __asm__ (
	   "fldd 0(%0),%%fr0\n"
	   : : "r" (bufptr), "m" (clear) : "%r0");

  return 0;
}

libm_hidden_def (__feholdexcept)
weak_alias (__feholdexcept, feholdexcept)
libm_hidden_weak (feholdexcept)
