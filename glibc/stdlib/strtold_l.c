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

#ifdef USE_WIDE_CHAR
# define STRING_TYPE	wchar_t
# define STRTOLD	wcstold_l
# define __STRTOLD	__wcstold_l
# define __STRTOD	__wcstod_l
#else
# define STRING_TYPE	char
# define STRTOLD	strtold_l
# define __STRTOLD	__strtold_l
# define __STRTOD	__strtod_l
#endif

#define INTERNAL(x) INTERNAL1(x)
#define INTERNAL1(x) __##x##_internal

extern double INTERNAL (__STRTOD) (const STRING_TYPE *, STRING_TYPE **,
				   int, locale_t);

/* There is no `long double' type, use the `double' implementations.  */
long double
INTERNAL (__STRTOLD) (const STRING_TYPE *nptr, STRING_TYPE **endptr,
		      int group, locale_t loc)
{
  return INTERNAL (__STRTOD) (nptr, endptr, group, loc);
}
#ifndef USE_WIDE_CHAR
libc_hidden_def (INTERNAL (__STRTOLD))
#endif

long double
weak_function
__STRTOLD (const STRING_TYPE *nptr, STRING_TYPE **endptr, locale_t loc)
{
  return INTERNAL (__STRTOD) (nptr, endptr, 0, loc);
}
#if defined _LIBC
libc_hidden_def (__STRTOLD)
libc_hidden_ver (__STRTOLD, STRTOLD)
#endif
weak_alias (__STRTOLD, STRTOLD)
