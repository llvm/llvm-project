/* Convert string representing a number to a _Float128 value, with locale.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <bits/types/locale_t.h>

/* Bring in potential typedef for _Float128 early for declaration below.  */
#include <bits/floatn.h>

#if __HAVE_FLOAT64X && !__HAVE_FLOAT64X_LONG_DOUBLE
# define strtof64x_l __hide_strtof64x_l
# define wcstof64x_l __hide_wcstof64x_l
#endif

extern _Float128 ____strtof128_l_internal (const char *, char **,
					   int, locale_t);

#define	FLOAT		_Float128
#define	FLT		FLT128
#ifdef USE_WIDE_CHAR
# define STRTOF		wcstof128_l
# define __STRTOF	__wcstof128_l
# define STRTOF_NAN	__wcstof128_nan
#else
# define STRTOF		strtof128_l
# define __STRTOF	__strtof128_l
# define STRTOF_NAN	__strtof128_nan
#endif
#define	MPN2FLOAT	__mpn_construct_float128
#define	FLOAT_HUGE_VAL	__builtin_huge_valf128 ()

#include <float128_private.h>

#include <stdlib/strtod_l.c>

#if __HAVE_FLOAT64X && !__HAVE_FLOAT64X_LONG_DOUBLE
# undef strtof64x_l
# undef wcstof64x_l
# ifdef USE_WIDE_CHAR
weak_alias (wcstof128_l, wcstof64x_l)
# else
weak_alias (strtof128_l, strtof64x_l)
# endif
#endif
