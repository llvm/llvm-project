/* Clear given exceptions in current floating-point environment.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv.h>
#include <fenv_libc.h>
#include <fpu_control.h>

int
feclearexcept (int excepts)
{
  int fpsr;

  /* Mask out unsupported bits/exceptions.  */
  excepts &= FE_ALL_EXCEPT;

  /* Read the complete control word.  */
  _FPU_GETFPSR (fpsr);

  /* Clear the relevant bits.  */
  fpsr &= ~(excepts | (excepts << CAUSE_SHIFT));

  /* Put the new data in effect.  */
  _FPU_SETFPSR (fpsr);

  return 0;
}
libm_hidden_def (feclearexcept)
