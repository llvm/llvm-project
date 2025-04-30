/* Store current floating-point environment.
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

#include <fenv.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <dl-procinfo.h>

int
__fegetenv (fenv_t *envp)
{
  __asm__ ("fnstenv %0" : "=m" (*envp));
  /* And load it right back since the processor changes the mask.
     Intel thought this opcode to be used in interrupt handlers which
     would block all exceptions.  */
  __asm__ ("fldenv %0" : : "m" (*envp));

  if (CPU_FEATURE_USABLE (SSE))
    __asm__ ("stmxcsr %0" : "=m" (envp->__eip));

  /* Success.  */
  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__fegetenv, __old_fegetenv)
compat_symbol (libm, __old_fegetenv, fegetenv, GLIBC_2_1);
#endif

libm_hidden_def (__fegetenv)
libm_hidden_ver (__fegetenv, fegetenv)
versioned_symbol (libm, __fegetenv, fegetenv, GLIBC_2_2);
