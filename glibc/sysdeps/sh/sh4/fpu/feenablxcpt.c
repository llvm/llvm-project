/* Enable floating-point exceptions.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Nobuhiro Iwamatsu <iwamatsu@nigauri.org>, 2012.

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

#include <fenv.h>
#include <fpu_control.h>

int
feenableexcept (int excepts)
{
  fpu_control_t temp, old_flag;

  /* Get current exceptions.  */
  _FPU_GETCW (temp);

  old_flag = (temp >> 5) & FE_ALL_EXCEPT;
  excepts &= FE_ALL_EXCEPT;

  temp |= excepts << 5;
  _FPU_SETCW (temp);

  return old_flag;
}
