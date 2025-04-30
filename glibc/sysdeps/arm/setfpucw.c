/* Set the FPU control word.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <fpu_control.h>
#include <arm-features.h>


void
__setfpucw (fpu_control_t set)
{
  fpu_control_t fpscr, new_fpscr;

  /* Do nothing if a VFP unit isn't present.  */
  if (!ARM_HAVE_VFP)
    return;

  _FPU_GETCW (fpscr);

  /* Preserve the reserved bits, and set the rest as the user
     specified (or the default, if the user gave zero).  */
  new_fpscr = fpscr & _FPU_RESERVED;
  new_fpscr |= set & ~_FPU_RESERVED;

  /* Write FPSCR if changed.  */
  if (new_fpscr != fpscr)
    _FPU_SETCW (fpscr);
}
