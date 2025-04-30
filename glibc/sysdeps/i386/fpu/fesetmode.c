/* Install given floating-point control modes.  i386 version.
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
#include <unistd.h>
#include <ldsodefs.h>
#include <dl-procinfo.h>

/* All exceptions, including the x86-specific "denormal operand"
   exception.  */
#define FE_ALL_EXCEPT_X86 (FE_ALL_EXCEPT | __FE_DENORM)

int
fesetmode (const femode_t *modep)
{
  fpu_control_t cw;
  if (modep == FE_DFL_MODE)
    cw = _FPU_DEFAULT;
  else
    cw = modep->__control_word;
  _FPU_SETCW (cw);
  if (CPU_FEATURE_USABLE (SSE))
    {
      unsigned int mxcsr;
      __asm__ ("stmxcsr %0" : "=m" (mxcsr));
      /* Preserve SSE exception flags but restore other state in
	 MXCSR.  */
      mxcsr &= FE_ALL_EXCEPT_X86;
      if (modep == FE_DFL_MODE)
	/* Default MXCSR state has all bits zero except for those
	   masking exceptions.  */
	mxcsr |= FE_ALL_EXCEPT_X86 << 7;
      else
	mxcsr |= modep->__mxcsr & ~FE_ALL_EXCEPT_X86;
      __asm__ ("ldmxcsr %0" : : "m" (mxcsr));
    }
  return 0;
}
