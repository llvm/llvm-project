/* Set current rounding direction.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <fenv_private.h>
#include <arm-features.h>


int
__fesetround (int round)
{
  /* FE_TONEAREST is the only supported rounding mode
     if a VFP unit isn't present.  */
  if (!ARM_HAVE_VFP)
    return (round == FE_TONEAREST) ? 0 : 1;

  if (round & ~_FPU_MASK_RM)
    return 1;

  libc_fesetround_vfp (round);
  return 0;
}
libm_hidden_def (__fesetround)
weak_alias (__fesetround, fesetround)
libm_hidden_weak (fesetround)
