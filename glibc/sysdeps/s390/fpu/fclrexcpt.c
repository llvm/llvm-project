/* Clear given exceptions in current floating-point environment.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
feclearexcept (int excepts)
{
  fexcept_t temp;

  /* Mask out unsupported bits/exceptions.  */
  excepts &= FE_ALL_EXCEPT;

  _FPU_GETCW (temp);
  /* Clear the relevant bits.  */
  temp &= ~(excepts << FPC_FLAGS_SHIFT);
  if ((temp & FPC_NOT_FPU_EXCEPTION) == 0)
    /* Bits 6, 7 of dxc-byte are zero,
       thus bits 0-5 of dxc-byte correspond to the flag-bits.
       Clear the relevant bits in flags and dxc-field.  */
    temp &= ~(excepts << FPC_DXC_SHIFT);

  /* Put the new data in effect.  */
  _FPU_SETCW (temp);

  /* Success.  */
  return 0;
}
libm_hidden_def (feclearexcept)
