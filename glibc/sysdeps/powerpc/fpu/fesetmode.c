/* Install given floating-point control modes.  PowerPC version.
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
#include <fpu_control.h>

int
fesetmode (const femode_t *modep)
{
  fenv_union_t old, new;

  /* Logic regarding enabled exceptions as in fesetenv.  */

  new.fenv = *modep;
  old.fenv = fegetenv_control ();
  new.l = (new.l & ~FPSCR_STATUS_MASK) | (old.l & FPSCR_STATUS_MASK);

  if (old.l == new.l)
    return 0;

  __TEST_AND_EXIT_NON_STOP (old.l, new.l);
  __TEST_AND_ENTER_NON_STOP (old.l, new.l);

  fesetenv_control (new.fenv);
  return 0;
}
