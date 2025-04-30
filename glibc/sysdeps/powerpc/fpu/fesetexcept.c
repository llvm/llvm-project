/* Set given exception flags.  PowerPC version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <fenv_libc.h>

int
fesetexcept (int excepts)
{
  fenv_union_t u, n;

  u.fenv = fegetenv_register ();
  n.l = (u.l
	 | (excepts & FPSCR_STICKY_BITS)
	 /* Turn FE_INVALID into FE_INVALID_SOFTWARE.  */
	 | (excepts >> ((31 - FPSCR_VX) - (31 - FPSCR_VXSOFT))
	    & FE_INVALID_SOFTWARE));
  if (n.l != u.l)
    {
      fesetenv_register (n.fenv);

      /* Deal with FE_INVALID_SOFTWARE not being implemented on some chips.  */
      if (excepts & FE_INVALID)
	feraiseexcept (FE_INVALID);
    }

  return 0;
}
