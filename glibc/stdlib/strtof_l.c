/* Convert string representing a number to float value, using given locale.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <bits/floatn.h>

#if __HAVE_FLOAT32 && !__HAVE_DISTINCT_FLOAT32
# define strtof32_l __hide_strtof32_l
# define wcstof32_l __hide_wcstof32_l
#endif

#include <locale.h>

extern float ____strtof_l_internal (const char *, char **, int, locale_t);

#define	FLOAT		float
#define	FLT		FLT
#ifdef USE_WIDE_CHAR
# define STRTOF		wcstof_l
# define __STRTOF	__wcstof_l
# define STRTOF_NAN	__wcstof_nan
#else
# define STRTOF		strtof_l
# define __STRTOF	__strtof_l
# define STRTOF_NAN	__strtof_nan
#endif
#define	MPN2FLOAT	__mpn_construct_float
#define	FLOAT_HUGE_VAL	HUGE_VALF

#include "strtod_l.c"

#if __HAVE_FLOAT32 && !__HAVE_DISTINCT_FLOAT32
# undef strtof32_l
# undef wcstof32_l
# ifdef USE_WIDE_CHAR
weak_alias (wcstof_l, wcstof32_l)
# else
weak_alias (strtof_l, strtof32_l)
# endif
#endif
