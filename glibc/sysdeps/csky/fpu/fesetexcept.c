/* Set given exception flags.
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
#include <fpu_control.h>
#include <fenv_libc.h>

int
fesetexcept (int excepts)
{
  fpu_control_t fpsr, new_fpsr;
  _FPU_GETFPSR (fpsr);
  new_fpsr = fpsr | ((excepts & FE_ALL_EXCEPT) << CAUSE_SHIFT);
  if (new_fpsr != fpsr)
    _FPU_SETFPSR (new_fpsr);

  return 0;
}
