/* Declarations for SVID math error handling compatibility.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#ifndef	_MATH_SVID_COMPAT_H
#define	_MATH_SVID_COMPAT_H	1

/* Support for various different standard error handling behaviors.  */
typedef enum
{
  _IEEE_ = -1,	/* According to IEEE 754/IEEE 854.  */
  _SVID_,	/* According to System V, release 4.  */
  _XOPEN_,	/* Nowadays also Unix98.  */
  _POSIX_,
  _ISOC_	/* Actually this is ISO C99.  */
} _LIB_VERSION_TYPE;

/* This variable can be changed at run-time to any of the values above to
   affect floating point error handling behavior (it may also be necessary
   to change the hardware FPU exception settings).  */
extern _LIB_VERSION_TYPE _LIB_VERSION;

/* In SVID error handling, `matherr' is called with this description
   of the exceptional condition.  */
struct exception
  {
    int type;
    char *name;
    double arg1;
    double arg2;
    double retval;
  };

extern int matherr (struct exception *__exc);
extern int __matherr (struct exception *__exc);

#define X_TLOSS	1.41484755040568800000e+16

/* Types of exceptions in the `type' field.  */
#define DOMAIN		1
#define SING		2
#define OVERFLOW	3
#define UNDERFLOW	4
#define TLOSS		5
#define PLOSS		6

/* SVID mode specifies returning this large value instead of infinity.  */
#define HUGE		3.40282347e+38F

/* The above definitions may be used in testcases.  The following code
   is only used in the implementation.  */

#ifdef _LIBC
/* fdlibm kernel function */
extern double __kernel_standard (double, double, int);
extern float __kernel_standard_f (float, float, int);
extern long double __kernel_standard_l (long double, long double, int);

# include <shlib-compat.h>
# define LIBM_SVID_COMPAT SHLIB_COMPAT (libm, GLIBC_2_0, GLIBC_2_27)
# if LIBM_SVID_COMPAT
compat_symbol_reference (libm, matherr, matherr, GLIBC_2_0);
compat_symbol_reference (libm, _LIB_VERSION, _LIB_VERSION, GLIBC_2_0);
# else
/* Except when building compat code, optimize out references to
   _LIB_VERSION and matherr.  */
#  define _LIB_VERSION _POSIX_
#  define matherr(EXC) ((void) (EXC), 0)
# endif
#endif

#endif /* math-svid-compat.h.  */
