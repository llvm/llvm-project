/* Raise given exceptions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <float.h>
#include <fpu_control.h>
#include <math.h>

int
__feraiseexcept (int excepts)
{
  if (excepts == 0)
    return 0;

  /* Raise exceptions represented by EXPECTS.  */

  if (excepts & FE_INEXACT)
  {
    double d = 1.0, x = 3.0;
    __asm__ __volatile__ ("fdiv %1, %0" : "+d" (d) : "d" (x));
  }

  if (excepts & FE_UNDERFLOW)
  {
    long double d = LDBL_MIN, x = 10;
    __asm__ __volatile__ ("fdiv %1, %0" : "+d" (d) : "d" (x));
  }

  if (excepts & FE_OVERFLOW)
  {
    long double d = LDBL_MAX;
    __asm__ __volatile__ ("fmul %0, %0" : "+d" (d) : "d" (d));
  }

  if (excepts & FE_DIVBYZERO)
  {
    double d = 1.0, x = 0.0;
    __asm__ __volatile__ ("fdiv %1, %0" : "+d" (d) : "d" (x));
  }

  if (excepts & FE_INVALID)
  {
    double d = HUGE_VAL, x = 0.0;
    __asm__ __volatile__ ("fmul %1, %0" : "+d" (d) : "d" (x));
  }

  {
    /* Restore flag fields.  */
    fpu_control_t cw;
    _FPU_GETCW (cw);
    cw |= (excepts & FE_ALL_EXCEPT);
    _FPU_SETCW (cw);
  }

  return 0;
}
libm_hidden_def (__feraiseexcept)
weak_alias (__feraiseexcept, feraiseexcept)
libm_hidden_weak (feraiseexcept)
