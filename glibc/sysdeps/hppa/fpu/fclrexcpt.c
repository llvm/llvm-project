/* Clear given exceptions in current floating-point environment.
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

int
feclearexcept (int excepts)
{
  union { unsigned long long l; unsigned int sw[2]; } s;

  /* Get the current status word. */
  __asm__ ("fstd %%fr0,0(%1)" : "=m" (s.l) : "r" (&s.l) : "%r0");
  /* Clear all the relevant bits. */
  s.sw[0] &= ~((excepts & FE_ALL_EXCEPT) << 27);
  __asm__ ("fldd 0(%0),%%fr0" : : "r" (&s.l), "m" (s.l) : "%r0");

  /* Success.  */
  return 0;
}
libm_hidden_def (feclearexcept)
