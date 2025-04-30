/* _Float128 overrides for float128 in ppc64le multiarch env.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_TYPE_MACROS_FLOAT128_PPC64_MULTI
#define _MATH_TYPE_MACROS_FLOAT128_PPC64_MULTI 1

#include_next <math-type-macros-float128.h>

#ifdef _F128_ENABLE_IFUNC

/* Include fenv.h now before turning off PLT bypass.  At
   minimum fereaiseexcept is used today.  */
#include <fenv.h>

#include <float128-ifunc-macros.h>

/* Ensure local redirects are always disabled by including
   math.h in the following manner.  */
#undef NO_MATH_REDIRECT
#define NO_MATH_REDIRECT
#include <math.h>
#undef NO_MATH_REDIRECT

/* Include complex prototypes now to enable redirecting of
   complex functions.  */
#include <complex.h>

/* Declare redirects for a function f which has a complex
   analogue.  That is, __ ## f ## f128 and __c ## f ## f128.  */
#define F128_C_REDIR(f) F128_REDIR (__c ## f ## f128); \
			F128_REDIR (__ ## f ## f128); \

/* Similar to F128_C_REDIR, declare the set of implementation
   redirects for the trigonometric family f for {a,}f{,h}
   and {a,}cf{,h} complex variants where f is sin/cos/tan.  */
#define F128_TRIG_REDIR(f) F128_C_REDIR (a ## f); \
			   F128_C_REDIR (a ## f ## h); \
			   F128_C_REDIR (f); \
			   F128_C_REDIR (f ## h);

F128_TRIG_REDIR (cos)
F128_TRIG_REDIR (sin)
F128_TRIG_REDIR (tan)

F128_C_REDIR (log);
F128_C_REDIR (log10);
F128_C_REDIR (exp);
F128_C_REDIR (sqrt);
F128_C_REDIR (pow);

F128_REDIR (__atan2f128)
F128_REDIR (__kernel_casinhf128);
F128_REDIR (__rintf128);
F128_REDIR (__floorf128);
F128_REDIR (__fabsf128);
F128_REDIR (__hypotf128);
F128_REDIR (__scalbnf128);
F128_REDIR (__scalblnf128);
F128_REDIR (__sincosf128);
F128_REDIR (__log1pf128);
F128_REDIR (__ilogbf128);
F128_REDIR (__ldexpf128);
F128_REDIR (__cargf128);
F128_REDIR (__cimagf128);
F128_REDIR (__crealf128);
F128_REDIR (__conjf128);
F128_REDIR (__cprojf128);
F128_REDIR (__cabsf128);
F128_REDIR (__fdimf128);
F128_REDIR (__fminf128);
F128_REDIR (__fmaxf128);
F128_REDIR (__fmodf128);
F128_REDIR (__llogbf128);
F128_REDIR (__log2f128);
F128_REDIR (__exp10f128);
F128_REDIR (__exp2f128);
F128_REDIR (__j0f128);
F128_REDIR (__j1f128);
F128_REDIR (__jnf128);
F128_REDIR (__y0f128);
F128_REDIR (__y1f128);
F128_REDIR (__ynf128);
F128_REDIR (__lgammaf128);
F128_REDIR_R (__lgammaf128, _r);
F128_REDIR (__tgammaf128);
F128_REDIR (__remainderf128);

/* This is ugly.  Some wrapper functions are neither prototyped nor declared
   uniformily (for various acceptable reasons).  A prototype is supplied
   to ensure they are appropriately ifunc'ed.  */
extern _Float128 __wrap_scalbnf128 (_Float128, int);
extern _Float128 __w_scalblnf128 (_Float128, long int);
extern _Float128 __w_log1pf128 (_Float128);
extern _Float128 __scalbf128 (_Float128, _Float128);
F128_REDIR (__scalbf128);
F128_REDIR (__wrap_scalbnf128);
F128_REDIR (__w_scalblnf128);
F128_REDIR (__w_log1pf128);

/* Include the redirects shared with math_private.h users.  */
#include <float128-ifunc-redirects.h>

#endif /* _F128_ENABLE_IFUNC */

#endif /*_MATH_TYPE_MACROS_FLOAT128_PPC64_MULTI */
