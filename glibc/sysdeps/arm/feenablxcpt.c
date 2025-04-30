/* Enable floating-point exceptions.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Philip Blundell <philb@gnu.org>, 2001.

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
#include <arm-features.h>


int
feenableexcept (int excepts)
{
  fpu_control_t fpscr, new_fpscr, updated_fpscr;

  /* Fail if a VFP unit isn't present.  */
  if (!ARM_HAVE_VFP)
    return -1;

  _FPU_GETCW (fpscr);
  excepts &= FE_ALL_EXCEPT;
  new_fpscr = fpscr | (excepts << FE_EXCEPT_SHIFT);

  if (new_fpscr != fpscr)
    {
      _FPU_SETCW (new_fpscr);

      /* Not all VFP architectures support trapping exceptions, so
	 test whether the relevant bits were set and fail if not.  */
      _FPU_GETCW (updated_fpscr);

      if (new_fpscr & ~updated_fpscr)
	return -1;
    }

  return (fpscr >> FE_EXCEPT_SHIFT) & FE_ALL_EXCEPT;
}
