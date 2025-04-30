/* Install given floating-point environment.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@tamu.edu>, 1997

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

#include <fenv_libc.h>

int
__fesetenv (const fenv_t *envp)
{
  unsigned long int fpcr;
  fenv_t env;

  /* Magic encoding of default values: high bit set (never possible for a
     user-space address) is not indirect.  And we don't even have to get
     rid of it since we mask things around just below.  */
  if ((long int) envp >= 0)
    env = *envp;
  else
    env = (unsigned long int) envp;

  /* Reset the rounding mode with the hardware fpcr.  Note that the following
     system call is an implied trap barrier for our modification.  */
  __asm__ __volatile__ ("excb; mf_fpcr %0" : "=f" (fpcr));
  fpcr = (fpcr & ~FPCR_ROUND_MASK) | (env & FPCR_ROUND_MASK);
  __asm__ __volatile__ ("mt_fpcr %0" : : "f" (fpcr));

  /* Reset the exception status and mask with the kernel's FP code.  */
  __ieee_set_fp_control (env & SWCR_ALL_MASK);

  /* Success.  */
  return 0;
}

#include <shlib-compat.h>
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_2)
strong_alias (__fesetenv, __old_fesetenv)
compat_symbol (libm, __old_fesetenv, fesetenv, GLIBC_2_1);
#endif

libm_hidden_def (__fesetenv)
libm_hidden_ver (__fesetenv, fesetenv)
versioned_symbol (libm, __fesetenv, fesetenv, GLIBC_2_2);
