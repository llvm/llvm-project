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

/* The actual implementation for all floating point sizes is in strtod.c.
   These macros tell it to produce the `long double' version, `strtold'.  */

#define FLOAT		long double
#define FLT		LDBL
#ifdef USE_WIDE_CHAR
# define STRTOF		wcstold_l
# define __STRTOF	__wcstold_l
# define STRTOF_NAN	__wcstold_nan
#else
# define STRTOF		strtold_l
# define __STRTOF	__strtold_l
# define STRTOF_NAN	__strtold_nan
#endif
#define MPN2FLOAT	__mpn_construct_long_double
#define FLOAT_HUGE_VAL	HUGE_VALL

#if __HAVE_FLOAT128 && !__HAVE_DISTINCT_FLOAT128
# define strtof128_l __hide_strtof128_l
# define wcstof128_l __hide_wcstof128_l
#endif

#if __HAVE_FLOAT64X_LONG_DOUBLE
# define strtof64x_l __hide_strtof64x_l
# define wcstof64x_l __hide_wcstof64x_l
#endif

#include <strtod_l.c>

#if __HAVE_FLOAT128 && !__HAVE_DISTINCT_FLOAT128
# undef strtof128_l
# undef wcstof128_l
# ifdef USE_WIDE_CHAR
weak_alias (wcstold_l, wcstof128_l)
# else
weak_alias (strtold_l, strtof128_l)
# endif
#endif

#if __HAVE_FLOAT64X_LONG_DOUBLE
# undef strtof64x_l
# undef wcstof64x_l
# ifdef USE_WIDE_CHAR
weak_alias (wcstold_l, wcstof64x_l)
# else
weak_alias (strtold_l, strtof64x_l)
# endif
#endif
