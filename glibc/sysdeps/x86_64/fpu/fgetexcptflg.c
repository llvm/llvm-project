/* Store current representation for exceptions.
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
fegetexceptflag (fexcept_t *flagp, int excepts)
{
  fexcept_t temp;
  unsigned int mxscr;

  /* Get the current exceptions for the x87 FPU and SSE unit.  */
  __asm__ ("fnstsw %0\n"
	   "stmxcsr %1" : "=m" (*&temp), "=m" (*&mxscr));

  *flagp = (temp | mxscr) & FE_ALL_EXCEPT & excepts;

  /* Success.  */
  return 0;
}
