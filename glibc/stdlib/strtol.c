/* Convert string representation of a number into an integer value.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <wchar.h>
#include <locale/localeinfo.h>

#ifndef UNSIGNED
# define UNSIGNED 0
# define INT LONG int
#else
# define INT unsigned LONG int
#endif

#if UNSIGNED
# ifdef USE_WIDE_CHAR
#  ifdef QUAD
#   define strtol wcstoull
#   define __strtol_l __wcstoull_l
#  else
#   define strtol wcstoul
#   define __strtol_l __wcstoul_l
#  endif
# else
#  ifdef QUAD
#   define strtol strtoull
#   define __strtol_l __strtoull_l
#  else
#   define strtol strtoul
#   define __strtol_l __strtoul_l
#  endif
# endif
#else
# ifdef USE_WIDE_CHAR
#  ifdef QUAD
#   define strtol wcstoll
#   define __strtol_l __wcstoll_l
#  else
#   define strtol wcstol
#   define __strtol_l __wcstol_l
#  endif
# else
#  ifdef QUAD
#   define strtol strtoll
#   define __strtol_l __strtoll_l
#  endif
# endif
#endif


/* If QUAD is defined, we are defining `strtoll' or `strtoull',
   operating on `long long int's.  */
#ifdef QUAD
# define LONG long long
#else
# define LONG long
#endif


#ifdef USE_WIDE_CHAR
# define STRING_TYPE wchar_t
#else
# define STRING_TYPE char
#endif


#define INTERNAL(X) INTERNAL1(X)
#define INTERNAL1(X) __##X##_internal

#define SYM__(X) SYM__1 (X)
#define SYM__1(X) __ ## X
#define __strtol SYM__ (strtol)


extern INT INTERNAL (__strtol_l) (const STRING_TYPE *, STRING_TYPE **, int,
				  int, locale_t);


INT
INTERNAL (strtol) (const STRING_TYPE *nptr, STRING_TYPE **endptr,
		   int base, int group)
{
  return INTERNAL (__strtol_l) (nptr, endptr, base, group, _NL_CURRENT_LOCALE);
}
libc_hidden_def (INTERNAL (strtol))


INT
__strtol (const STRING_TYPE *nptr, STRING_TYPE **endptr, int base)
{
  return INTERNAL (__strtol_l) (nptr, endptr, base, 0, _NL_CURRENT_LOCALE);
}
weak_alias (__strtol, strtol)
libc_hidden_weak (strtol)
