/* Definitions of libc internal inline math functions implemented
   by the m68881/2.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _MATHIMPL_H
#define _MATHIMPL_H

/* This file contains the definitions of the inline math functions that
   are only used internally inside libm, not visible to the user.  */

#define __MATH_INLINE __extern_inline

/* This is used when defining the functions themselves.  Define them with
   __ names, and with `static inline' instead of `extern inline' so the
   bodies will always be used, never an external function call.
   Note: GCC 6 objects to __attribute__ ((__leaf__)) on static functions.  */
#define __m81_u(x)             __CONCAT(__,x)
#define __m81_inline           static __inline
#define __m81_nth(fn)          __NTH (fn)

/* Define a math function.  */
#define __m81_defun(rettype, func, args, attrs)	\
  __m81_inline rettype attrs			\
  __m81_nth (__m81_u(func) args)

/* Define the three variants of a math function that has a direct
   implementation in the m68k fpu.  FUNC is the name for C (which will be
   suffixed with f and l for the float and long double version, resp).  OP
   is the name of the fpu operation (without leading f).  */

# define __inline_mathop(func, op, attrs)			\
  __inline_mathop1(double, func, op, attrs)			\
  __inline_mathop1(float, __CONCAT(func,f), op, attrs)		\
  __inline_mathop1(long double, __CONCAT(func,l), op, attrs)

#define __inline_mathop1(float_type,func, op, attrs)			      \
  __m81_defun (float_type, func, (float_type __mathop_x), attrs)	      \
  {									      \
    float_type __result;						      \
    __asm __volatile__ ("f" __STRING(op) "%.x %1, %0"			      \
			: "=f" (__result) : "f" (__mathop_x));		      \
    return __result;							      \
  }

__inline_mathop(__atan, atan,)
__inline_mathop(__cos, cos,)
__inline_mathop(__sin, sin,)
__inline_mathop(__tan, tan,)
__inline_mathop(__tanh, tanh,)
__inline_mathop(__fabs, abs, __attribute__ ((__const__)))

__inline_mathop(__rint, int,)
__inline_mathop(__expm1, etoxm1,)
__inline_mathop(__log1p, lognp1,)

__inline_mathop(__significand, getman,)

__inline_mathop(__trunc, intrz, __attribute__ ((__const__)))


/* This macro contains the definition for the rest of the inline
   functions, using FLOAT_TYPE as the domain type and M as a macro
   that adds the suffix for the function names.  */

#define __inline_functions(float_type, m)				  \
__m81_defun (float_type, m(__floor), (float_type __x),			  \
	     __attribute__ ((__const__)))				  \
{									  \
  float_type __result;							  \
  unsigned long int __ctrl_reg;						  \
  __asm __volatile__ ("fmove%.l %!, %0" : "=dm" (__ctrl_reg));		  \
  /* Set rounding towards negative infinity.  */			  \
  __asm __volatile__ ("fmove%.l %0, %!" : /* No outputs.  */		  \
		      : "dmi" ((__ctrl_reg & ~0x10) | 0x20));		  \
  /* Convert X to an integer, using -Inf rounding.  */			  \
  __asm __volatile__ ("fint%.x %1, %0" : "=f" (__result) : "f" (__x));	  \
  /* Restore the previous rounding mode.  */				  \
  __asm __volatile__ ("fmove%.l %0, %!" : /* No outputs.  */		  \
		      : "dmi" (__ctrl_reg));				  \
  return __result;							  \
}									  \
									  \
__m81_defun (float_type, m(__ceil), (float_type __x),			  \
	     __attribute__ ((__const__)))				  \
{									  \
  float_type __result;							  \
  unsigned long int __ctrl_reg;						  \
  __asm __volatile__ ("fmove%.l %!, %0" : "=dm" (__ctrl_reg));		  \
  /* Set rounding towards positive infinity.  */			  \
  __asm __volatile__ ("fmove%.l %0, %!" : /* No outputs.  */		  \
		      : "dmi" (__ctrl_reg | 0x30));			  \
  /* Convert X to an integer, using +Inf rounding.  */			  \
  __asm __volatile__ ("fint%.x %1, %0" : "=f" (__result) : "f" (__x));	  \
  /* Restore the previous rounding mode.  */				  \
  __asm __volatile__ ("fmove%.l %0, %!" : /* No outputs.  */		  \
		      : "dmi" (__ctrl_reg));				  \
  return __result;							  \
}

#define __CONCAT_d(arg) arg
#define __CONCAT_f(arg) arg ## f
#define __CONCAT_l(arg) arg ## l
__inline_functions(double, __CONCAT_d)
__inline_functions(float, __CONCAT_f)
__inline_functions(long double, __CONCAT_l)
#undef __inline_functions

# define __inline_functions(float_type, m)				  \
__m81_defun (int, m(__isinf), (float_type __value),			  \
	     __attribute__ ((__const__)))				  \
{									  \
  /* There is no branch-condition for infinity,				  \
     so we must extract and examine the condition codes manually.  */	  \
  unsigned long int __fpsr;						  \
  __asm ("ftst%.x %1\n"							  \
	 "fmove%.l %/fpsr, %0" : "=dm" (__fpsr) : "f" (__value));	  \
  return (__fpsr & (2 << 24)) ? (__fpsr & (8 << 24) ? -1 : 1) : 0;	  \
}									  \
									  \
__m81_defun (int, m(__finite), (float_type __value),			  \
	     __attribute__ ((__const__)))				  \
{									  \
  /* There is no branch-condition for infinity, so we must extract and	  \
     examine the condition codes manually.  */				  \
  unsigned long int __fpsr;						  \
  __asm ("ftst%.x %1\n"							  \
	 "fmove%.l %/fpsr, %0" : "=dm" (__fpsr) : "f" (__value));	  \
  return (__fpsr & (3 << 24)) == 0;					  \
}									  \
									  \
__m81_defun (float_type, m(__scalbn),					  \
	     (float_type __x, int __n),)				  \
{									  \
  float_type __result;							  \
  __asm __volatile__  ("fscale%.l %1, %0" : "=f" (__result)		  \
		       : "dmi" (__n), "0" (__x));			  \
  return __result;							  \
}

__inline_functions(double, __CONCAT_d)
__inline_functions(float, __CONCAT_f)
__inline_functions(long double, __CONCAT_l)
#undef __inline_functions

# define __inline_functions(float_type, m)				  \
__m81_defun (int, m(__isnan), (float_type __value),			  \
	     __attribute__ ((__const__)))			  	  \
{									  \
  char __result;							  \
  __asm ("ftst%.x %1\n"							  \
	 "fsun %0" : "=dm" (__result) : "f" (__value));			  \
  return __result;							  \
}

__inline_functions(double, __CONCAT_d)
__inline_functions(float, __CONCAT_f)
__inline_functions(long double, __CONCAT_l)
#undef __inline_functions

# define __inline_functions(float_type, m)				  \
__m81_defun (float_type, m(__scalbln),					  \
	     (float_type __x, long int __n),)				  \
{									  \
  return m(__scalbn) (__x, __n);					  \
}									  \
									  \
__m81_defun (float_type, m(__nearbyint), (float_type __x),)		  \
{									  \
  float_type __result;							  \
  unsigned long int __ctrl_reg;						  \
  __asm __volatile__ ("fmove%.l %!, %0" : "=dm" (__ctrl_reg));		  \
  /* Temporarily disable the inexact exception.  */			  \
  __asm __volatile__ ("fmove%.l %0, %!" : /* No outputs.  */		  \
		      : "dmi" (__ctrl_reg & ~0x200));			  \
  __asm __volatile__ ("fint%.x %1, %0" : "=f" (__result) : "f" (__x));	  \
  __asm __volatile__ ("fmove%.l %0, %!" : /* No outputs.  */		  \
		      : "dmi" (__ctrl_reg));				  \
  return __result;							  \
}									  \
									  \
__m81_defun (long int, m(__lrint), (float_type __x),)			  \
{									  \
  long int __result;							  \
  __asm __volatile__ ("fmove%.l %1, %0" : "=dm" (__result) : "f" (__x));  \
  return __result;							  \
}

__inline_functions (double, __CONCAT_d)
__inline_functions (float, __CONCAT_f)
__inline_functions (long double, __CONCAT_l)
#undef __inline_functions

#define __inline_functions(float_type, m)				\
__m81_inline void							\
__m81_nth (__m81_u(m(__sincos))						\
	   (float_type __x, float_type *__sinx, float_type *__cosx))	\
{									\
  __asm __volatile__ ("fsincos%.x %2,%1:%0"				\
		      : "=f" (*__sinx), "=f" (*__cosx) : "f" (__x));	\
}

__inline_functions (double, __CONCAT_d)
__inline_functions (float, __CONCAT_f)
__inline_functions (long double, __CONCAT_l)
#undef __inline_functions

#undef __CONCAT_d
#undef __CONCAT_f
#undef __CONCAT_l

/* Define the three variants of a math function that has a direct
   implementation in the m68k fpu.  FUNC is the name for C (which will be
   suffixed with f and l for the float and long double version, resp).  OP
   is the name of the fpu operation (without leading f).  */

#define __inline_mathop(func, op, attrs)			\
  __inline_mathop1(double, func, op, attrs)			\
  __inline_mathop1(float, __CONCAT(func,f), op, attrs)		\
  __inline_mathop1(long double, __CONCAT(func,l), op, attrs)

#define __inline_mathop1(float_type,func, op, attrs)			      \
  __m81_defun (float_type, func, (float_type __mathop_x), attrs)	      \
  {									      \
    float_type __result;						      \
    __asm __volatile__ ("f" __STRING(op) "%.x %1, %0"			      \
			: "=f" (__result) : "f" (__mathop_x));		      \
    return __result;							      \
  }

__inline_mathop	(__ieee754_acos, acos,)
__inline_mathop	(__ieee754_asin, asin,)
__inline_mathop	(__ieee754_cosh, cosh,)
__inline_mathop	(__ieee754_sinh, sinh,)
__inline_mathop	(__ieee754_exp, etox,)
__inline_mathop	(__ieee754_exp2, twotox,)
__inline_mathop	(__ieee754_exp10, tentox,)
__inline_mathop	(__ieee754_log10, log10,)
__inline_mathop	(__ieee754_log2, log2,)
__inline_mathop	(__ieee754_log, logn,)
__inline_mathop	(__ieee754_sqrt, sqrt,)
__inline_mathop	(__ieee754_atanh, atanh,)

__m81_defun (double, __ieee754_remainder, (double __x, double __y),)
{
  double __result;
  __asm ("frem%.x %1, %0" : "=f" (__result) : "f" (__y), "0" (__x));
  return __result;
}

__m81_defun (float, __ieee754_remainderf, (float __x, float __y),)
{
  float __result;
  __asm ("frem%.x %1, %0" : "=f" (__result) : "f" (__y), "0" (__x));
  return __result;
}

__m81_defun (long double,
	     __ieee754_remainderl, (long double __x, long double __y),)
{
  long double __result;
  __asm ("frem%.x %1, %0" : "=f" (__result) : "f" (__y), "0" (__x));
  return __result;
}

__m81_defun (double, __ieee754_fmod, (double __x, double __y),)
{
  double __result;
  __asm ("fmod%.x %1, %0" : "=f" (__result) : "f" (__y), "0" (__x));
  return __result;
}

__m81_defun (float, __ieee754_fmodf, (float __x, float __y),)
{
  float __result;
  __asm ("fmod%.x %1, %0" : "=f" (__result) : "f" (__y), "0" (__x));
  return __result;
}

__m81_defun (long double,
	     __ieee754_fmodl, (long double __x, long double __y),)
{
  long double __result;
  __asm ("fmod%.x %1, %0" : "=f" (__result) : "f" (__y), "0" (__x));
  return __result;
}

/* Get the m68881 condition codes, to quickly check multiple conditions.  */
static __inline__ unsigned long
__m81_test (long double __val)
{
  unsigned long __fpsr;
  __asm ("ftst%.x %1; fmove%.l %/fpsr,%0" : "=dm" (__fpsr) : "f" (__val));
  return __fpsr;
}

/* Bit values returned by __m81_test.  */
#define __M81_COND_NAN  (1 << 24)
#define __M81_COND_INF  (2 << 24)
#define __M81_COND_ZERO (4 << 24)
#define __M81_COND_NEG  (8 << 24)

#endif /* _MATHIMPL_H  */
