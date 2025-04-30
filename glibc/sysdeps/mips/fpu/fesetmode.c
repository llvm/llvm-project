/* Install given floating-point control modes.  MIPS version.
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

#include <fenv.h>
#include <fpu_control.h>

#define FCSR_STATUS 0xfe83f07c

int
fesetmode (const femode_t *modep)
{
  fpu_control_t cw;

  _FPU_GETCW (cw);
  cw &= FCSR_STATUS;
  if (modep == FE_DFL_MODE)
    cw |= _FPU_DEFAULT;
  else
    cw |= *modep & ~FCSR_STATUS;
  _FPU_SETCW (cw);

  return 0;
}
