/* Compare at most N wide characters of two strings without taking care
   for the case.
   Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <wchar.h>
#include <wctype.h>

#ifndef weak_alias
# define __wcsncasecmp wcsncasecmp
# define TOLOWER(Ch) towlower (Ch)
#else
# ifdef USE_IN_EXTENDED_LOCALE_MODEL
#  define __wcsncasecmp __wcsncasecmp_l
#  define TOLOWER(Ch) __towlower_l ((Ch), loc)
# else
#  define TOLOWER(Ch) towlower (Ch)
# endif
#endif

#ifdef USE_IN_EXTENDED_LOCALE_MODEL
# define LOCALE_PARAM , locale_t loc
#else
# define LOCALE_PARAM
#endif

/* Compare no more than N wide characters of S1 and S2,
   ignoring case, returning less than, equal to or
   greater than zero if S1 is lexicographically less
   than, equal to or greater than S2.  */
int
__wcsncasecmp (const wchar_t *s1, const wchar_t *s2, size_t n LOCALE_PARAM)
{
  wint_t c1, c2;

  if (s1 == s2 || n == 0)
    return 0;

  do
    {
      c1 = (wint_t) TOLOWER (*s1++);
      c2 = (wint_t) TOLOWER (*s2++);
      if (c1 == L'\0' || c1 != c2)
	return c1 - c2;
    } while (--n > 0);

  return c1 - c2;
}
#ifndef __wcsncasecmp
weak_alias (__wcsncasecmp, wcsncasecmp)
#endif
