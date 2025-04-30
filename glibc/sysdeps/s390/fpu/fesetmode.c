/* Install given floating-point control modes.  S/390 version.
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
#include <fenv_libc.h>

#define FPC_STATUS (FPC_FLAGS_MASK | FPC_DXC_MASK)

int
fesetmode (const femode_t *modep)
{
  fpu_control_t fpc;

  _FPU_GETCW (fpc);
  fpc &= FPC_STATUS;
  if (modep == FE_DFL_MODE)
    fpc |= _FPU_DEFAULT;
  else
    fpc |= *modep & ~FPC_STATUS;
  _FPU_SETCW (fpc);

  return 0;
}
