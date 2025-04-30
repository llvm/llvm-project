/* Copyright (C) 1997-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv.h>
#include <fpu_control.h>
#include <float.h>

int
__feraiseexcept (int excepts)
{
  int fpsr;
  const float fp_zero = 0.0;
  const float fp_one = 1.0;
  const float fp_max = FLT_MAX;
  const float fp_min = FLT_MIN;
  const float fp_1e32 = 1.0e32f;
  const float fp_two = 2.0;
  const float fp_three = 3.0;

  /* Raise exceptions represented by EXCEPTS.  But we must raise only
     one signal at a time.  It is important that if the OVERFLOW or
     UNDERFLOW exception and the inexact exception are given at the
     same time, the OVERFLOW or UNDERFLOW exception precedes the
     INEXACT exception.

     After each exception we read from the FPSR, to force the
     exception to be raised immediately.  */

  if (FE_INVALID & excepts)
    __asm__ __volatile__ (
			  "ldr	s0, %1\n\t"
			  "fdiv	s0, s0, s0\n\t"
			  "mrs	%0, fpsr" : "=r" (fpsr)
			  : "m" (fp_zero)
			  : "d0");

  if (FE_DIVBYZERO & excepts)
    __asm__ __volatile__ (
			  "ldr	s0, %1\n\t"
			  "ldr	s1, %2\n\t"
			  "fdiv	s0, s0, s1\n\t"
			  "mrs	%0, fpsr" : "=r" (fpsr)
			  : "m" (fp_one), "m" (fp_zero)
			  : "d0", "d1");

  if (FE_OVERFLOW & excepts)
    /* There's no way to raise overflow without also raising inexact.  */
    __asm__ __volatile__ (
			  "ldr	s0, %1\n\t"
			  "ldr	s1, %2\n\t"
			  "fadd s0, s0, s1\n\t"
			  "mrs	%0, fpsr" : "=r" (fpsr)
			  : "m" (fp_max), "m" (fp_1e32)
			  : "d0", "d1");

  if (FE_UNDERFLOW & excepts)
    __asm__ __volatile__ (
			  "ldr	s0, %1\n\t"
			  "ldr	s1, %2\n\t"
			  "fdiv s0, s0, s1\n\t"
			  "mrs	%0, fpsr" : "=r" (fpsr)
			  : "m" (fp_min), "m" (fp_three)
			  : "d0", "d1");

  if (FE_INEXACT & excepts)
    __asm__ __volatile__ (
			  "ldr	s0, %1\n\t"
			  "ldr	s1, %2\n\t"
			  "fdiv s0, s0, s1\n\t"
			  "mrs	%0, fpsr" : "=r" (fpsr)
			  : "m" (fp_two), "m" (fp_three)
			  : "d0", "d1");

  return 0;
}
libm_hidden_def (__feraiseexcept)
weak_alias (__feraiseexcept, feraiseexcept)
libm_hidden_weak (feraiseexcept)
