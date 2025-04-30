/* Set FP exception mask and rounding mode.
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

#include <fpu_control.h>
#include <fenv_libc.h>


#define convert_bit(M, F, T)		\
    ((T) < (F)				\
     ? ((M) / ((F) / (T))) & (T)	\
     : ((M) & (F)) * ((T) / (F)))


void
__setfpucw (fpu_control_t fpu_control)
{
  unsigned long fpcr, swcr, fc = (int)fpu_control;

  /* ??? If this was a real external interface we'd want to read the current
     exception state with __ieee_get_fp_control.  But this is an internal
     function only called at process startup, so there's no point in trying
     to preserve exceptions that cannot have been raised yet.  Indeed, this
     entire function is likely to be one big nop unless the user overrides
     the default __fpu_control variable.  */

  /* Convert the rounding mode from fpu_control.h format.  */
  const unsigned long conv_rnd
    = (  (FE_TOWARDZERO << (_FPU_RC_ZERO >> 8))
       | (FE_DOWNWARD << (_FPU_RC_DOWN >> 8))
       | (FE_TONEAREST << (_FPU_RC_NEAREST >> 8))
       | (FE_UPWARD << (_FPU_RC_UP >> 8)));

  fpcr = ((conv_rnd >> ((fc >> 8) & 3)) & 3) << FPCR_ROUND_SHIFT;

  /* Convert the exception mask from fpu_control.h format.  */
  swcr  = convert_bit (~fc, _FPU_MASK_IM, FE_INVALID >> SWCR_ENABLE_SHIFT);
  swcr |= convert_bit (~fc, _FPU_MASK_DM, FE_UNDERFLOW >> SWCR_ENABLE_SHIFT);
  swcr |= convert_bit (~fc, _FPU_MASK_ZM, FE_DIVBYZERO >> SWCR_ENABLE_SHIFT);
  swcr |= convert_bit (~fc, _FPU_MASK_OM, FE_OVERFLOW >> SWCR_ENABLE_SHIFT);
  swcr |= convert_bit (~fc, _FPU_MASK_PM, FE_INEXACT >> SWCR_ENABLE_SHIFT);

  /* Install everything.  */
  __fpu_control = fc;
  asm volatile ("mt_fpcr %0" : : "f"(fpcr));
  __ieee_set_fp_control(swcr);
}
