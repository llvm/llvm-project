/* Copyright (C) 1996-2021 Free Software Foundation, Inc.

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

#include <math.h>
#include <get-rounding-mode.h>
#include <stdint.h>
#include <math-barriers.h>
#include <libm-alias-double.h>

# define IREG_SIZE 64

# ifdef __ILP32__
#  define OREG_SIZE 32
# else
#  define OREG_SIZE 64
# endif

# define IREGS "d"

#if OREG_SIZE == 32
# define OREGS "w"
#else
# define OREGS "x"
#endif


long int
__lrint (double x)
{

#if IREG_SIZE == 64 && OREG_SIZE == 32
  long int result;

  if (__builtin_fabs (x) > INT32_MAX)
    {
      /* Converting large values to a 32 bit int may cause the frintx/fcvtza
	 sequence to set both FE_INVALID and FE_INEXACT.  To avoid this
	 check the rounding mode and do a single instruction with the
	 appropriate rounding mode.  */

      switch (get_rounding_mode ())
	{
	case FE_TONEAREST:
	  asm volatile ("fcvtns" "\t%" OREGS "0, %" IREGS "1"
			: "=r" (result) : "w" (x));
	  break;
	case FE_UPWARD:
	  asm volatile ("fcvtps" "\t%" OREGS "0, %" IREGS "1"
			: "=r" (result) : "w" (x));
	  break;
	case FE_DOWNWARD:
	  asm volatile ("fcvtms" "\t%" OREGS "0, %" IREGS "1"
			: "=r" (result) : "w" (x));
	  break;
	case FE_TOWARDZERO:
	default:
	  asm volatile ("fcvtzs" "\t%" OREGS "0, %" IREGS "1"
			: "=r" (result) : "w" (x));
	}
      return result;
    }
#endif

  double r =  __builtin_rint (x);

  /* Prevent gcc from calling lrint directly when compiled with
     -fno-math-errno by inserting a barrier.  */

  math_opt_barrier (r);
  return r;
}

libm_alias_double (__lrint, lrint)
