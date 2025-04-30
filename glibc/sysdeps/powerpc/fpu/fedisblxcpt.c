/* Disable floating-point exceptions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Geoffrey Keating <geoffk@geoffk.org>, 2000.

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
fedisableexcept (int excepts)
{
  fenv_union_t fe, curr;
  int result, new;

  /* Get current exception mask to return.  */
  fe.fenv = curr.fenv = fegetenv_control ();
  result = fenv_reg_to_exceptions (fe.l);

  if ((excepts & FE_ALL_INVALID) == FE_ALL_INVALID)
    excepts = (excepts | FE_INVALID) & ~ FE_ALL_INVALID;

  new = fenv_exceptions_to_reg (excepts);

  if (fenv_reg_to_exceptions (new) != excepts)
    return -1;

  /* Sets the new exception mask.  */
  fe.l &= ~new;

  if (fe.l != curr.l)
    fesetenv_control (fe.fenv);

  __TEST_AND_ENTER_NON_STOP (-1ULL, fe.l);

  return result;
}
