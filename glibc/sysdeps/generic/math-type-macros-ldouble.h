/* Helper macros for long double variants of type generic functions of libm.
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

#ifndef _MATH_TYPE_MACROS_LDOUBLE
#define _MATH_TYPE_MACROS_LDOUBLE

#define M_LIT(c) c ## L
#define M_MLIT(c) c ## l
#define M_PFX LDBL
#define M_SUF(c) c ## l
#define FLOAT long double
#define CFLOAT _Complex long double
#define M_STRTO_NAN __strtold_nan

#include <libm-alias-ldouble.h>
#include <math-nan-payload-ldouble.h>

#ifndef declare_mgen_alias
# define declare_mgen_alias(from, to) libm_alias_ldouble (from, to)
#endif

#ifndef declare_mgen_alias_r
# define declare_mgen_alias_r(from, to) libm_alias_ldouble_r (from, to, _r)
#endif

/* Supply the generic macros.  */
#include <math-type-macros.h>

/* Do not use the type-generic wrapper templates if compatibility with
   SVID error handling is needed.  */
#include <math/math-svid-compat.h>
#define __USE_WRAPPER_TEMPLATE !LIBM_SVID_COMPAT

#endif
