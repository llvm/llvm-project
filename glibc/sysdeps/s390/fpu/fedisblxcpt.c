/* Disable floating-point exceptions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Denis Joseph Barrow (djbarrow@de.ibm.com).

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
fedisableexcept (int excepts)
{
  fexcept_t temp, old_exc, new_flags;

  _FPU_GETCW (temp);
  old_exc = (temp & FPC_EXCEPTION_MASK) >> FPC_EXCEPTION_MASK_SHIFT;
  new_flags = (temp & (~((excepts & FE_ALL_EXCEPT) << FPC_EXCEPTION_MASK_SHIFT)));
  _FPU_SETCW (new_flags);

  return old_exc;
}
