/* Test exception in current environment.
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
fetestexcept (int excepts)
{
  union { unsigned long long l; unsigned int sw[2]; } s;

  /* Get the current status word. */
  __asm__ ("fstd %%fr0,0(%1)	\n\t"
           "fldd 0(%1),%%fr0	\n\t"
	   : "=m" (s.l) : "r" (&s.l));

  return (s.sw[0] >> 27) & excepts & FE_ALL_EXCEPT;
}
libm_hidden_def (fetestexcept)
