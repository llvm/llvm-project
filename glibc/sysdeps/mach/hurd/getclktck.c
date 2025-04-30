/* Return run-time value of CLK_TCK for Hurd.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <time.h>

/* Return frequency of `times'.
   Since Mach reports CPU times in microseconds, we always use 1 million.  */
int
__getclktck (void)
{
  return 1000000;
}

/* Before glibc 2.2, the Hurd actually did this differently, so we
   need to keep a compatibility symbol.  */

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_1_1, GLIBC_2_2)
compat_symbol (libc, __getclktck, __libc_clk_tck, GLIBC_2_1_1);
#endif
