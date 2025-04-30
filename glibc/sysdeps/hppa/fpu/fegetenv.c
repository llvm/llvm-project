/* Store current floating-point environment.
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
__fegetenv (fenv_t *envp)
{
  unsigned long long buf[4], *bufptr = buf;

  __asm__ (
	   "fstd,ma %%fr0,8(%1)	\n\t"
	   "fldd -8(%1),%%fr0	\n\t"
	   : "=m" (buf), "+r" (bufptr) : : "%r0");
  memcpy(envp, buf, sizeof (*envp));
  return 0;
}
libm_hidden_def (__fegetenv)
weak_alias (__fegetenv, fegetenv)
libm_hidden_weak (fegetenv)
