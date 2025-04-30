/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <stdlib.h>
#include <wchar.h>

/* The actual implementation for all floating point sizes is in strtod.c.
   These macros tell it to produce the `long double' version, `strtold'.  */

#define FLOAT		long double
#define FLT		LDBL
#ifdef USE_WIDE_CHAR
extern long double ____new_wcstold_l (const wchar_t *, wchar_t **, locale_t);
# define STRTOF		__new_wcstold_l
# define __STRTOF	____new_wcstold_l
# define ____STRTOF_INTERNAL ____wcstold_l_internal
# define STRTOF_NAN	__wcstold_nan
#else
extern long double ____new_strtold_l (const char *, char **, locale_t);
# define STRTOF		__new_strtold_l
# define __STRTOF	____new_strtold_l
# define ____STRTOF_INTERNAL ____strtold_l_internal
# define STRTOF_NAN	__strtold_nan
#endif
extern __typeof (__STRTOF) STRTOF;
libc_hidden_proto (__STRTOF)
libc_hidden_proto (STRTOF)
#define MPN2FLOAT	__mpn_construct_long_double
#define FLOAT_HUGE_VAL	HUGE_VALL

#include <strtod_l.c>

#ifdef __LONG_DOUBLE_MATH_OPTIONAL
# include <math_ldbl_opt.h>
# ifdef USE_WIDE_CHAR
weak_alias (____new_wcstold_l, ___new_wcstold_l);
long_double_symbol (libc, ___new_wcstold_l, wcstold_l);
long_double_symbol (libc, ____new_wcstold_l, __wcstold_l);
# else
weak_alias (____new_strtold_l, ___new_strtold_l);
long_double_symbol (libc, ___new_strtold_l, strtold_l);
long_double_symbol (libc, ____new_strtold_l, __strtold_l);
# endif
#endif
