/* Read decimal floating point numbers.
   This file is part of the GNU C Library.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1995.

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

#ifdef FLOAT
# define BUILD_DOUBLE 0
#else
# define BUILD_DOUBLE 1
#endif

#if BUILD_DOUBLE
# if __HAVE_FLOAT64 && !__HAVE_DISTINCT_FLOAT64
#  define strtof64 __hide_strtof64
#  define wcstof64 __hide_wcstof64
# endif
# if __HAVE_FLOAT32X && !__HAVE_DISTINCT_FLOAT32X
#  define strtof32x __hide_strtof32x
#  define wcstof32x __hide_wcstof32x
# endif
#endif

#include <stdlib.h>
#include <wchar.h>
#include <locale/localeinfo.h>


#ifndef FLOAT
# include <math_ldbl_opt.h>
# define FLOAT double
# ifdef USE_WIDE_CHAR
#  define STRTOF wcstod
#  define STRTOF_L __wcstod_l
# else
#  define STRTOF strtod
#  define STRTOF_L __strtod_l
# endif
#endif

#ifdef USE_WIDE_CHAR
# include <wctype.h>
# define STRING_TYPE wchar_t
#else
# define STRING_TYPE char
#endif

#define INTERNAL(x) INTERNAL1(x)
#define INTERNAL1(x) __##x##_internal


FLOAT
INTERNAL (STRTOF) (const STRING_TYPE *nptr, STRING_TYPE **endptr, int group)
{
  return INTERNAL(STRTOF_L) (nptr, endptr, group, _NL_CURRENT_LOCALE);
}
#if defined _LIBC
libc_hidden_def (INTERNAL (STRTOF))
#endif


FLOAT
#ifdef weak_function
weak_function
#endif
STRTOF (const STRING_TYPE *nptr, STRING_TYPE **endptr)
{
  return INTERNAL(STRTOF_L) (nptr, endptr, 0, _NL_CURRENT_LOCALE);
}
#if defined _LIBC
libc_hidden_def (STRTOF)
#endif

#ifdef LONG_DOUBLE_COMPAT
# if LONG_DOUBLE_COMPAT(libc, GLIBC_2_0)
#  ifdef USE_WIDE_CHAR
compat_symbol (libc, wcstod, wcstold, GLIBC_2_0);
compat_symbol (libc, __wcstod_internal, __wcstold_internal, GLIBC_2_0);
#  else
compat_symbol (libc, strtod, strtold, GLIBC_2_0);
compat_symbol (libc, __strtod_internal, __strtold_internal, GLIBC_2_0);
#  endif
# endif
#endif

#if BUILD_DOUBLE
# if __HAVE_FLOAT64 && !__HAVE_DISTINCT_FLOAT64
#  undef strtof64
#  undef wcstof64
#  ifdef USE_WIDE_CHAR
weak_alias (wcstod, wcstof64)
#  else
weak_alias (strtod, strtof64)
#  endif
# endif
# if __HAVE_FLOAT32X && !__HAVE_DISTINCT_FLOAT32X
#  undef strtof32x
#  undef wcstof32x
#  ifdef USE_WIDE_CHAR
weak_alias (wcstod, wcstof32x)
#  else
weak_alias (strtod, strtof32x)
#  endif
# endif
#endif
