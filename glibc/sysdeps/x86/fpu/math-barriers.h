/* Control when floating-point expressions are evaluated.  x86 version.
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

#ifndef X86_MATH_BARRIERS_H
#define X86_MATH_BARRIERS_H 1

#if defined(__SSE2_MATH__) && !defined(__clang__)
# define math_opt_barrier(x)						\
  ({ __typeof(x) __x;							\
     if (sizeof (x) <= sizeof (double)					\
	|| __builtin_types_compatible_p (__typeof (x), _Float128))	\
       __asm ("" : "=x" (__x) : "0" (x));				\
     else								\
       __asm ("" : "=t" (__x) : "0" (x));				\
     __x; })
# define math_force_eval(x)						\
  do {									\
    if (sizeof (x) <= sizeof (double)					\
	|| __builtin_types_compatible_p (__typeof (x), _Float128))	\
      __asm __volatile ("" : : "x" (x));				\
    else								\
      __asm __volatile ("" : : "f" (x));				\
  } while (0)
#else
# define math_opt_barrier(x)						\
  ({ __typeof (x) __x;							\
     if (__builtin_types_compatible_p (__typeof (x), _Float128))	\
       {								\
	 __x = (x);							\
	 __asm ("" : "+m" (__x));					\
       }								\
     else								\
       __asm ("" : "=t" (__x) : "0" (x));				\
     __x; })
# define math_force_eval(x)						\
  do {									\
    __typeof (x) __x = (x);						\
    if (sizeof (x) <= sizeof (double)					\
	|| __builtin_types_compatible_p (__typeof (x), _Float128))	\
      __asm __volatile ("" : : "m" (__x));				\
    else								\
      __asm __volatile ("" : : "f" (__x));				\
  } while (0)
#endif

#endif
