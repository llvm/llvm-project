/* Set floating-point environment exception handling.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Hartvig Ekner <hartvige@mips.com>, 2002.

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
#include <fpu_control.h>

int
fesetexceptflag (const fexcept_t *flagp, int excepts)
{
  fpu_control_t temp;

  /* Get the current exceptions.  */
  _FPU_GETCW (temp);

  /* Make sure the flags we want restored are legal.  */
  excepts &= FE_ALL_EXCEPT;

  /* Now clear the bits called for, and copy them in from flagp. Note that
     we ignore all non-flag bits from *flagp, so they don't matter.  */
  temp = (temp & ~excepts) | (*flagp & excepts);

  _FPU_SETCW (temp);

  /* Success.  */
  return 0;
}
