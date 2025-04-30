/* Check for underflow and force underflow exceptions.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_UNDERFLOW_H
#define _MATH_UNDERFLOW_H	1

#include <float.h>
#include <math.h>

#include <math-barriers.h>

#define fabs_tg(x) __MATH_TG ((x), (__typeof (x)) __builtin_fabs, (x))

/* These must be function-like macros because some __MATH_TG
   implementations macro-expand the function-name argument before
   concatenating a suffix to it.  */
#define min_of_type_f() FLT_MIN
#define min_of_type_() DBL_MIN
#define min_of_type_l() LDBL_MIN
#define min_of_type_f128() FLT128_MIN

#define min_of_type(x) __MATH_TG ((x), (__typeof (x)) min_of_type_, ())

/* If X (which is not a NaN) is subnormal, force an underflow
   exception.  */
#define math_check_force_underflow(x)				\
  do								\
    {								\
      __typeof (x) force_underflow_tmp = (x);			\
      if (fabs_tg (force_underflow_tmp)				\
	  < min_of_type (force_underflow_tmp))			\
	{							\
	  __typeof (force_underflow_tmp) force_underflow_tmp2	\
	    = force_underflow_tmp * force_underflow_tmp;	\
	  math_force_eval (force_underflow_tmp2);		\
	}							\
    }								\
  while (0)
/* Likewise, but X is also known to be nonnegative.  */
#define math_check_force_underflow_nonneg(x)			\
  do								\
    {								\
      __typeof (x) force_underflow_tmp = (x);			\
      if (force_underflow_tmp					\
	  < min_of_type (force_underflow_tmp))			\
	{							\
	  __typeof (force_underflow_tmp) force_underflow_tmp2	\
	    = force_underflow_tmp * force_underflow_tmp;	\
	  math_force_eval (force_underflow_tmp2);		\
	}							\
    }								\
  while (0)
/* Likewise, for both real and imaginary parts of a complex
   result.  */
#define math_check_force_underflow_complex(x)				\
  do									\
    {									\
      __typeof (x) force_underflow_complex_tmp = (x);			\
      math_check_force_underflow (__real__ force_underflow_complex_tmp); \
      math_check_force_underflow (__imag__ force_underflow_complex_tmp); \
    }									\
  while (0)

#endif /* math-underflow.h */
