/* Software floating-point emulation.
   Helper routine for _Qp_* routines.
   Simulate exceptions using double arithmetics.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek (jj@ultra.linux.cz).

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

#include <float.h>
#include <math.h>
#include <assert.h>
#include "soft-fp.h"

void __Qp_handle_exceptions(int exceptions)
{
  if (exceptions & FP_EX_INVALID)
    {
      float f = 0.0;
      __asm__ __volatile__ ("fdivs %0, %0, %0" : "+f" (f));
    }
  if (exceptions & FP_EX_DIVZERO)
    {
      float f = 1.0, g = 0.0;
      __asm__ __volatile__ ("fdivs %0, %1, %0"
			    : "+f" (f)
			    : "f" (g));
    }
  if (exceptions & FP_EX_OVERFLOW)
    {
      float f = FLT_MAX;
      __asm__ __volatile__("fmuls %0, %0, %0" : "+f" (f));
      exceptions &= ~FP_EX_INEXACT;
    }
  if (exceptions & FP_EX_UNDERFLOW)
    {
      float f = FLT_MIN;
      __asm__ __volatile__("fmuls %0, %0, %0" : "+f" (f));
      exceptions &= ~FP_EX_INEXACT;
    }
  if (exceptions & FP_EX_INEXACT)
    {
      double d = 1.0, e = M_PI;
      __asm__ __volatile__ ("fdivd %0, %1, %0"
			    : "+f" (d)
			    : "f" (e));
    }
}
