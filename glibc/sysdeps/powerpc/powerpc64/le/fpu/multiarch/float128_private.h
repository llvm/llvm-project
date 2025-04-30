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

#ifndef _FLOAT128_PRIVATE_PPC64LE
#define _FLOAT128_PRIVATE_PPC64LE 1

#ifndef _F128_ENABLE_IFUNC
/* multiarch is not supported.  Do nothing and pass through.  */
#include_next <float128_private.h>
#else

/* Include fenv.h now before turning off libm_hidden_proto.
   At minimum, fereaiseexcept is needed. */
#include <fenv.h>

/* The PLT bypass trick used by libm_hidden_proto uses asm-renames.
   If gcc detects a second rename to a different function, it will
   emit errors.  */
#undef libm_hidden_proto
#define libm_hidden_proto(f)

/* Always disable redirects.  We supply these uniquely later on.  */
#undef NO_MATH_REDIRECT
#define NO_MATH_REDIRECT
#include <math.h>
#undef NO_MATH_REDIRECT

#include_next <float128_private.h>

#include <float128-ifunc-macros.h>

/* Declare these now.  These prototyes are not included
   in any header.  */
extern __typeof (cosf128) __ieee754_cosf128;
extern __typeof (asinhf128) __ieee754_asinhf128;

F128_REDIR (__ieee754_asinhf128)
F128_REDIR (__ieee754_cosf128)
F128_REDIR (__asinhf128)
F128_REDIR (__atanf128)
F128_REDIR (__cbrtf128)
F128_REDIR (__ceilf128)
F128_REDIR (__cosf128)
F128_REDIR (__erfcf128)
F128_REDIR (__erff128)
F128_REDIR (__expf128)
F128_REDIR (__expm1f128)
F128_REDIR (__fabsf128)
F128_REDIR (__fdimf128)
F128_REDIR (__floorf128)
F128_REDIR (__fmaf128)
F128_REDIR (__fmaxf128)
F128_REDIR (__fminf128)
F128_REDIR (__frexpf128)
F128_REDIR (__ldexpf128)
F128_REDIR (__llrintf128)
F128_REDIR (__llroundf128)
F128_REDIR (__log1pf128)
F128_REDIR (__logbf128)
F128_REDIR (__logf128)
F128_REDIR (__lrintf128)
F128_REDIR (__lroundf128)
F128_REDIR (__modff128)
F128_REDIR (__nearbyintf128)
F128_REDIR (__remquof128)
F128_REDIR (__rintf128)
F128_REDIR (__roundevenf128)
F128_REDIR (__roundf128)
F128_REDIR (__scalblnf128)
F128_REDIR (__scalbnf128)
F128_REDIR (__sincosf128)
F128_REDIR (__sinf128)
F128_REDIR (__sqrtf128)
F128_REDIR (__tanhf128)
F128_REDIR (__tanf128)
F128_REDIR (__truncf128)
F128_REDIR (__lgamma_productf128)

#include <float128-ifunc-redirects-mp.h>
#include <float128-ifunc-redirects.h>

#endif /* _F128_ENABLE_IFUNC */

#endif /* _FLOAT128_PRIVATE_PPC64LE */
