/* Round to integer generic implementation.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#ifndef _ROUND_TO_INTEGER_H
#define _ROUND_TO_INTEGER_H

#include <fenv_private.h>

enum round_mode
{
  CEIL,
  FLOOR,
  ROUND,
  TRUNC,
  NEARBYINT,
  RINT
};

static inline fenv_t
set_fenv_mode (enum round_mode mode)
{
  fenv_t fe = 0;
  if (mode != RINT)
    /* Save current FPU rounding mode and inexact state.  */
    fe = fegetenv_register ();

  switch (mode)
  {
  case CEIL:
    __fesetround_inline_nocheck (FE_UPWARD);
    break;
  case FLOOR:
    __fesetround_inline_nocheck (FE_DOWNWARD);
    break;
  case TRUNC:
  case ROUND:
    __fesetround_inline_nocheck (FE_TOWARDZERO);
    break;
  case NEARBYINT:
    /*  Disable FE_INEXACT exception  */
    reset_fpscr_bit (FPSCR_XE);
    break;
  case RINT:
    break;
  }
  return fe;
}

static inline void
reset_fenv_mode (fenv_t fe, enum round_mode mode)
{
  switch (mode)
  {
  default:
    __builtin_mtfsf (0xff, fe);
    break;
  case RINT:
    break;
  }
}

static inline float
round_to_integer_float (enum round_mode mode, float x)
{
  /* Ensure sNaN input is converted to qNaN.  */
  if (__glibc_unlikely (isnan (x)))
    return x + x;

  if (fabs (x) > 0x1p+23)
    return x;

  float r = x;

  fenv_t fe = set_fenv_mode (mode);
  if (x > 0.0)
    {
      /* IEEE 1003.1 round function.  IEEE specifies "round to the nearest
	 integer value, rounding halfway cases away from zero, regardless of
	 the current rounding mode."  However PowerPC Architecture defines
	 "Round to Nearest" as "Choose the best approximation. In case of a
	 tie, choose the one that is even (least significant bit o).".
	 So we can't use the PowerPC "Round to Nearest" mode. Instead we set
	 "Round toward Zero" mode and round by adding +-0.5 before rounding
	 to the integer value.  */
      if (mode == ROUND)
	r += 0.5f;
      r += 0x1p+23;
      r -= 0x1p+23;
      r = fabs (r);
    }
  else if (x < 0.0)
    {
      if (mode == ROUND)
	r -= 0.5f;
      r -= 0x1p+23;
      r += 0x1p+23;
      r = -fabs (r);
    }
  reset_fenv_mode (fe, mode);

  return r;
}

static inline double
round_to_integer_double (enum round_mode mode, double x)
{
  /* Ensure sNaN input is converted to qNaN.  */
  if (__glibc_unlikely (isnan (x)))
    return x + x;

  if (fabs (x) > 0x1p+52)
    return x;

  double r = x;

  /* Save current FPU rounding mode and inexact state.  */
  fenv_t fe = set_fenv_mode (mode);
  if (x > 0.0)
    {
      if (mode == ROUND)
	r += 0.5;
      r += 0x1p+52;
      r -= 0x1p+52;
      r = fabs (r);
    }
  else if (x < 0.0)
    {
      if (mode == ROUND)
	r -= 0.5;
      r -= 0x1p+52;
      r += 0x1p+52;
      r = -fabs (r);
    }
  reset_fenv_mode (fe, mode);

  return r;
}

#endif
