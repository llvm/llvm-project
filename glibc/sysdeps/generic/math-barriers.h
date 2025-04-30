/* Control when floating-point expressions are evaluated.  Generic version.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_BARRIERS_H
#define _MATH_BARRIERS_H	1

/* math_opt_barrier evaluates and returns its floating-point argument
   and ensures that the evaluation of any expression using the result
   of math_opt_barrier is not moved before the call.  math_force_eval
   ensures that its floating-point argument is evaluated for its side
   effects even if its value is apparently unused, and that the
   evaluation of its argument is not moved after the call.  Both these
   macros are used to ensure the correct ordering of floating-point
   expression evaluations with respect to accesses to the
   floating-point environment.  */

#define math_opt_barrier(x)					\
  ({ __typeof (x) __x = (x); __asm ("" : "+m" (__x)); __x; })
#define math_force_eval(x)						\
  ({ __typeof (x) __x = (x); __asm __volatile__ ("" : : "m" (__x)); })

#endif /* math-barriers.h */
