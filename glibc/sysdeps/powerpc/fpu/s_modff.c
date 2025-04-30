/* Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

/* ISA 2.07 provides fast GPR to FP instruction (mfvsr{d,wz}) which make
   generic implementation faster.  Also disables for old ISAs that do not
   have ceil/floor instructions.  */
#if defined(_ARCH_PWR8) || !defined(_ARCH_PWR5X)
# include <sysdeps/ieee754/flt-32/s_modff.c>
#else
# include <math.h>
# include <libm-alias-float.h>

float
__modff (float x, float *iptr)
{
  if (__builtin_isinff (x))
    {
      *iptr = x;
      return copysignf (0.0, x);
    }
  else if (__builtin_isnanf (x))
    {
      *iptr = NAN;
      return NAN;
    }

  if (x >= 0.0)
    {
      *iptr = floorf (x);
      return copysignf (x - *iptr, x);
    }
  else
    {
      *iptr = ceilf (x);
      return copysignf (x - *iptr, x);
    }
}
# ifndef __modff
libm_alias_float (__modf, modf)
# endif
#endif
