/* Helper macros for type generic function implementations within libm.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_TYPE_MACROS
#define _MATH_TYPE_MACROS

/* Each type imports a header which is expected to
   define:

   M_LIT(x)   - Paste the type specific suffix onto the constant x.
   M_MLIT(x)  - Paste the type specific suffix used by the macro
		constants in math.h, i.e M_PI or M_PIl.
   M_PFX      - The prefixed used by float.h macros like FLT_MANT_DIG.
   M_SUF(x)   - Paste the the type specific suffix used by functions
		i.e expf expl exp.
   FLOAT      - Resolves to the C typename of M_TYPE.
   CFLOAT     - Resolves to the complex typename of M_TYPE.
   M_STRTO_NAN - Resolves to the internal libc function which
		converts a string into the appropriate FLOAT nan
		value.

  declare_mgen_alias(from,to)
      This exposes the appropriate symbol(s) for a
      function f of type FLOAT.

  declare_mgen_alias_r(from,to)
      This exposes the appropriate symbol(s) for a
      function f_r of type FLOAT.

  SET_NAN_PAYLOAD(flt, mant)
      Set the NaN payload bits of the variable FLT of type FLOAT to
      the mantissa MANT.  */

#ifndef M_PFX
# error "M_PFX must be defined."
#endif
#ifndef M_LIT
# error "M_LIT must be defined."
#endif
#ifndef M_MLIT
# error "M_MLIT must be defined."
#endif
#ifndef M_SUF
# error "M_SUF must be defined."
#endif
#ifndef FLOAT
# error "FLOAT must be defined."
#endif
#ifndef CFLOAT
# error "CFLOAT must be defined."
#endif
#ifndef declare_mgen_alias
# error "declare_mgen_alias must be defined."
#endif
#ifndef declare_mgen_alias_r
# error "declare_mgen_alias_r must be defined."
#endif
#ifndef SET_NAN_PAYLOAD
# error "SET_NAN_PAYLOAD must be defined."
#endif

#ifndef declare_mgen_finite_alias_x
#define declare_mgen_finite_alias_x(from, to)   \
  libm_alias_finite (from, to)
#endif

#ifndef declare_mgen_finite_alias_s
# define declare_mgen_finite_alias_s(from,to)	\
  declare_mgen_finite_alias_x (from, to)
#endif

#ifndef declare_mgen_finite_alias
# define declare_mgen_finite_alias(from, to)	\
  declare_mgen_finite_alias_s (M_SUF (from), M_SUF (to))
#endif

#define __M_CONCAT(a,b) a ## b
#define __M_CONCATX(a,b) __M_CONCAT(a,b)

#define M_NAN M_SUF (__builtin_nan) ("")
#define M_MIN_EXP __M_CONCATX (M_PFX, _MIN_EXP)
#define M_MAX_EXP __M_CONCATX (M_PFX, _MAX_EXP)
#define M_MIN __M_CONCATX (M_PFX, _MIN)
#define M_MAX __M_CONCATX (M_PFX, _MAX)
#define M_MANT_DIG __M_CONCATX (M_PFX, _MANT_DIG)
#define M_HUGE_VAL (M_SUF (__builtin_huge_val) ())

/* Helper macros for commonly used functions.  */
#define M_COPYSIGN M_SUF (copysign)
#define M_FABS M_SUF (fabs)
#define M_SINCOS M_SUF (__sincos)
#define M_SCALBN M_SUF (__scalbn)
#define M_LOG1P M_SUF (__log1p)

#define M_ATAN2 M_SUF (__ieee754_atan2)
#define M_COSH M_SUF (__ieee754_cosh)
#define M_EXP M_SUF (__ieee754_exp)
#define M_HYPOT M_SUF (__ieee754_hypot)
#define M_LOG M_SUF (__ieee754_log)
#define M_SINH M_SUF (__ieee754_sinh)
#define M_SQRT M_SUF (sqrt)

/* Needed to evaluate M_MANT_DIG below.  */
#include <float.h>
#include <libm-alias-finite.h>

/* Use a special epsilon value for IBM long double
   to avoid spurious overflows/underflows.  */
#if M_MANT_DIG != 106
# define M_EPSILON __M_CONCATX (M_PFX, _EPSILON)
#else
# define M_EPSILON M_LIT (0x1p-106)
#endif

/* Enable overloading of function name to assist reuse.  */
#ifndef M_DECL_FUNC
# define M_DECL_FUNC(f) M_SUF (f)
#endif

#endif /* _MATH_TYPE_MACROS */
