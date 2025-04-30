/* Compatibility macros for old and new Alpha complex float ABI.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

/* The behaviour of complex float changed between GCC 3.3 and 3.4.

   In 3.3 and before (below, complex version 1, or "c1"), complex float
   values were packed into one floating point register.

   In 3.4 and later (below, complex version 2, or "c2"), GCC changed to
   follow the official Tru64 ABI, which passes the components of a complex
   as separate parameters.  */

typedef union { double d; _Complex float cf; } c1_compat;
# define c1_cfloat_decl(x)	double x
# define c1_cfloat_real(x)	__real__ c1_cfloat_value (x)
# define c1_cfloat_imag(x)	__imag__ c1_cfloat_value (x)
# define c1_cfloat_value(x)	(((c1_compat *)(void *)&x)->cf)
# define c1_cfloat_rettype	double
# define c1_cfloat_return(x)	({ c1_compat _; _.cf = (x); _.d; })

# define c2_cfloat_decl(x)	_Complex float x
# define c2_cfloat_real(x)	__real__ x
# define c2_cfloat_imag(x)	__imag__ x
# define c2_cfloat_value(x)	x
# define c2_cfloat_rettype	_Complex float
# define c2_cfloat_return(x)	x

/* Get the proper symbol versions defined for each function.  */

#include <shlib-compat.h>
#include <libm-alias-float.h>

#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_3_4)
#define cfloat_versions_compat(func) \
  compat_symbol (libm, __c1_##func, func, GLIBC_2_1)
#else
#define cfloat_versions_compat(func)
#endif

#define cfloat_versions(func) \
  cfloat_versions_compat(func##f); \
  versioned_symbol (libm, __c2_##func##f, func##f, GLIBC_2_3_4); \
  extern typeof(__c2_##func##f) __##func##f attribute_hidden; \
  strong_alias (__c2_##func##f, __##func##f); \
  libm_alias_float_other (__##func, func)
