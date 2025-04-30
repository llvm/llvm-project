/* Raise given exceptions.
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

#include <fpu_control.h>
#include <fenv.h>
#include <float.h>
#include <arm-features.h>


int
__feraiseexcept (int excepts)
{
  /* Fail if a VFP unit isn't present unless nothing needs to be done.  */
  if (!ARM_HAVE_VFP)
    return (excepts != 0);
  else
    {
      fpu_control_t fpscr;
      const float fp_zero = 0.0, fp_one = 1.0, fp_max = FLT_MAX,
                  fp_min = FLT_MIN, fp_1e32 = 1.0e32f, fp_two = 2.0,
		  fp_three = 3.0;

      /* Raise exceptions represented by EXPECTS.  But we must raise only
	 one signal at a time.  It is important that if the overflow/underflow
	 exception and the inexact exception are given at the same time,
	 the overflow/underflow exception follows the inexact exception.  After
	 each exception we read from the fpscr, to force the exception to be
	 raised immediately.  */

      /* There are additional complications because this file may be compiled
         without VFP support enabled, and we also can't assume that the
	 assembler has VFP instructions enabled. To get around this we use the
	 generic coprocessor mnemonics and avoid asking GCC to put float values
	 in VFP registers.  */

      /* First: invalid exception.  */
      if (FE_INVALID & excepts)
	__asm__ __volatile__ (
	  "ldc p10, cr0, %1\n\t"                        /* flds s0, %1  */
	  "cdp p10, 8, cr0, cr0, cr0, 0\n\t"            /* fdivs s0, s0, s0  */
	  "mrc p10, 7, %0, cr1, cr0, 0" : "=r" (fpscr)  /* fmrx %0, fpscr  */
			                : "m" (fp_zero)
					: "s0");

      /* Next: division by zero.  */
      if (FE_DIVBYZERO & excepts)
	__asm__ __volatile__ (
	  "ldc p10, cr0, %1\n\t"                        /* flds s0, %1  */
	  "ldcl p10, cr0, %2\n\t"                       /* flds s1, %2  */
	  "cdp p10, 8, cr0, cr0, cr0, 1\n\t"            /* fdivs s0, s0, s1  */
	  "mrc p10, 7, %0, cr1, cr0, 0" : "=r" (fpscr)  /* fmrx %0, fpscr  */
			                : "m" (fp_one), "m" (fp_zero)
					: "s0", "s1");

      /* Next: overflow.  */
      if (FE_OVERFLOW & excepts)
	/* There's no way to raise overflow without also raising inexact.  */
	__asm__ __volatile__ (
	  "ldc p10, cr0, %1\n\t"                        /* flds s0, %1  */
	  "ldcl p10, cr0, %2\n\t"                       /* flds s1, %2  */
	  "cdp p10, 3, cr0, cr0, cr0, 1\n\t"            /* fadds s0, s0, s1  */
	  "mrc p10, 7, %0, cr1, cr0, 0" : "=r" (fpscr)  /* fmrx %0, fpscr  */
			                : "m" (fp_max), "m" (fp_1e32)
					: "s0", "s1");

      /* Next: underflow.  */
      if (FE_UNDERFLOW & excepts)
	__asm__ __volatile__ (
	  "ldc p10, cr0, %1\n\t"                        /* flds s0, %1  */
	  "ldcl p10, cr0, %2\n\t"                       /* flds s1, %2  */
	  "cdp p10, 8, cr0, cr0, cr0, 1\n\t"            /* fdivs s0, s0, s1  */
	  "mrc p10, 7, %0, cr1, cr0, 0" : "=r" (fpscr)  /* fmrx %0, fpscr  */
			                : "m" (fp_min), "m" (fp_three)
					: "s0", "s1");

      /* Last: inexact.  */
      if (FE_INEXACT & excepts)
	__asm__ __volatile__ (
	  "ldc p10, cr0, %1\n\t"                        /* flds s0, %1  */
	  "ldcl p10, cr0, %2\n\t"                       /* flds s1, %2  */
	  "cdp p10, 8, cr0, cr0, cr0, 1\n\t"            /* fdivs s0, s0, s1  */
	  "mrc p10, 7, %0, cr1, cr0, 0" : "=r" (fpscr)  /* fmrx %0, fpscr  */
			                : "m" (fp_two), "m" (fp_three)
					: "s0", "s1");

      /* Success.  */
      return 0;
    }
}
libm_hidden_def (__feraiseexcept)
weak_alias (__feraiseexcept, feraiseexcept)
libm_hidden_weak (feraiseexcept)
