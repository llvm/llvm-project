/* Install given floating-point environment.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jes Sorensen <Jes.Sorensen@cern.ch>, 2000

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
__fesetenv (const fenv_t *envp)
{
  fenv_t env;

  /*
     This stinks!
     Magic encoding of default values: bit 62+63 set (which will never
     happen for a user-space address) means it's not indirect.
  */
  if (((fenv_t) envp >> 62) == 0x03)
    env = (fenv_t) envp & 0x3fffffffffffffff;
  else
    env = *envp;

  __asm__ __volatile__ ("mov.m ar.fpsr=%0;;" :: "r" (env));

  return 0;
}
libm_hidden_def (__fesetenv)
weak_alias (__fesetenv, fesetenv)
libm_hidden_weak (fesetenv)
