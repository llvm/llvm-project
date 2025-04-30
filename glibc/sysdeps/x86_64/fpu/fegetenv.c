/* Store current floating-point environment.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

int
__fegetenv (fenv_t *envp)
{
  __asm__ ("fnstenv %0\n"
	   /* fnstenv changes the exception mask, so load back the
	      stored environment.  */
	   "fldenv %0\n"
	   "stmxcsr %1" : "=m" (*envp), "=m" (envp->__mxcsr));

  /* Success.  */
  return 0;
}
libm_hidden_def (__fegetenv)
weak_alias (__fegetenv, fegetenv)
libm_hidden_weak (fegetenv)
