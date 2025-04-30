/* Install given floating-point control modes.  ARM version.
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
#include <arm-features.h>

/* NZCV flags, QC bit, IDC bit and bits for IEEE exception status.  */
#define FPU_STATUS_BITS 0xf800009f

int
fesetmode (const femode_t *modep)
{
  fpu_control_t fpscr, new_fpscr;

  if (!ARM_HAVE_VFP)
    /* Nothing to do.  */
    return 0;

  _FPU_GETCW (fpscr);
  if (modep == FE_DFL_MODE)
    new_fpscr = (fpscr & (_FPU_RESERVED | FPU_STATUS_BITS)) | _FPU_DEFAULT;
  else
    new_fpscr = (fpscr & FPU_STATUS_BITS) | (*modep & ~FPU_STATUS_BITS);

  if (((new_fpscr ^ fpscr) & ~_FPU_MASK_NZCV) != 0)
    _FPU_SETCW (new_fpscr);

  return 0;
}
