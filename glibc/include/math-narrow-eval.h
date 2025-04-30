/* Narrow floating-point values to their semantic type.
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

#ifndef _MATH_NARROW_EVAL_H
#define _MATH_NARROW_EVAL_H	1

#include <float.h>

/* math_narrow_eval reduces its floating-point argument to the range
   and precision of its semantic type.  (The original evaluation may
   still occur with excess range and precision, so the result may be
   affected by double rounding.)  */
#if FLT_EVAL_METHOD == 0
# define math_narrow_eval(x) (x)
#else
# if FLT_EVAL_METHOD == 1
#  define excess_precision(type) __builtin_types_compatible_p (type, float)
# else
#  define excess_precision(type) (__builtin_types_compatible_p (type, float) \
				  || __builtin_types_compatible_p (type, \
								   double))
# endif
# define math_narrow_eval(x)					\
  ({								\
    __typeof (x) math_narrow_eval_tmp = (x);			\
    if (excess_precision (__typeof (math_narrow_eval_tmp)))	\
      __asm__ ("" : "+m" (math_narrow_eval_tmp));		\
    math_narrow_eval_tmp;					\
   })
#endif

#endif /* math-narrow-eval.h */
