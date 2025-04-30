/* Test exception in current environment.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
fetestexcept (int excepts)
{
  fenv_union_t u;

  /* Get the current state.  */
  u.fenv = fegetenv_register ();

  /* The FE_INVALID bit is dealt with correctly by the hardware, so we can
     just:  */
  return u.l & excepts;
}
libm_hidden_def (fetestexcept)
