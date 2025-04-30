/* Convert string representing a number to integer value, using given locale.
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


#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef _LIBC
# define USE_NUMBER_GROUPING
# define HAVE_LIMITS_H
#endif

#include <ctype.h>
#include <errno.h>
#ifndef __set_errno
# define __set_errno(Val) errno = (Val)
#endif

#ifdef HAVE_LIMITS_H
# include <limits.h>
#endif

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <stdint.h>
#include <bits/wordsize.h>

#ifdef USE_NUMBER_GROUPING
# include "../locale/localeinfo.h"
#endif

/* Nonzero if we are defining `strtoul' or `strtoull', operating on
   unsigned integers.  */
#ifndef UNSIGNED
# define UNSIGNED 0
# define INT LONG int
#else
# define INT unsigned LONG int
#endif

/* Determine the name.  */
#if UNSIGNED
# ifdef USE_WIDE_CHAR
#  ifdef QUAD
#   define strtol_l wcstoull_l
#  else
#   define strtol_l wcstoul_l
#  endif
# else
#  ifdef QUAD
#   define strtol_l strtoull_l
#  else
#   define strtol_l strtoul_l
#  endif
# endif
#else
# ifdef USE_WIDE_CHAR
#  ifdef QUAD
#   define strtol_l wcstoll_l
#  else
#   define strtol_l wcstol_l
#  endif
# else
#  ifdef QUAD
#   define strtol_l strtoll_l
#  else
#   define strtol_l strtol_l
#  endif
# endif
#endif

#define __strtol_l __strtol_l2(strtol_l)
#define __strtol_l2(name) __strtol_l3(name)
#define __strtol_l3(name) __##name


/* If QUAD is defined, we are defining `strtoll' or `strtoull',
   operating on `long long int's.  */
#ifdef QUAD
# define LONG long long
# define STRTOL_LONG_MIN LONG_LONG_MIN
# define STRTOL_LONG_MAX LONG_LONG_MAX
# define STRTOL_ULONG_MAX ULONG_LONG_MAX
#else
# define LONG long

# ifndef ULONG_MAX
#  define ULONG_MAX ((unsigned long int) ~(unsigned long int) 0)
# endif
# ifndef LONG_MAX
#  define LONG_MAX ((long int) (ULONG_MAX >> 1))
# endif
# define STRTOL_LONG_MIN LONG_MIN
# define STRTOL_LONG_MAX LONG_MAX
# define STRTOL_ULONG_MAX ULONG_MAX
#endif


/* We use this code for the extended locale handling where the
   function gets as an additional argument the locale which has to be
   used.  To access the values we have to redefine the _NL_CURRENT and
   _NL_CURRENT_WORD macros.  */
#undef _NL_CURRENT
#define _NL_CURRENT(category, item) \
  (current->values[_NL_ITEM_INDEX (item)].string)
#undef _NL_CURRENT_WORD
#define _NL_CURRENT_WORD(category, item) \
  ((uint32_t) current->values[_NL_ITEM_INDEX (item)].word)

#if defined _LIBC || defined HAVE_WCHAR_H
# include <wchar.h>
#endif

#ifdef USE_WIDE_CHAR
# include <wctype.h>
# define L_(Ch) L##Ch
# define UCHAR_TYPE wint_t
# define STRING_TYPE wchar_t
# define ISSPACE(Ch) __iswspace_l ((Ch), loc)
# define ISALPHA(Ch) __iswalpha_l ((Ch), _nl_C_locobj_ptr)
# define TOUPPER(Ch) __towupper_l ((Ch), _nl_C_locobj_ptr)
#else
# if defined _LIBC \
   || defined STDC_HEADERS || (!defined isascii && !defined HAVE_ISASCII)
#  define IN_CTYPE_DOMAIN(c) 1
# else
#  define IN_CTYPE_DOMAIN(c) isascii(c)
# endif
# define L_(Ch) Ch
# define UCHAR_TYPE unsigned char
# define STRING_TYPE char
# define ISSPACE(Ch) __isspace_l ((Ch), loc)
# define ISALPHA(Ch) __isalpha_l ((Ch), _nl_C_locobj_ptr)
# define TOUPPER(Ch) __toupper_l ((Ch), _nl_C_locobj_ptr)
#endif

#define INTERNAL(X) INTERNAL1(X)
#define INTERNAL1(X) __##X##_internal
#define WEAKNAME(X) WEAKNAME1(X)

#ifdef USE_NUMBER_GROUPING
/* This file defines a function to check for correct grouping.  */
# include "grouping.h"
#endif


/* Define tables of maximum values and remainders in order to detect
   overflow.  Do this at compile-time in order to avoid the runtime
   overhead of the division.  */
extern const unsigned long __strtol_ul_max_tab[] attribute_hidden;
extern const unsigned char __strtol_ul_rem_tab[] attribute_hidden;
#if defined(QUAD) && __WORDSIZE == 32
extern const unsigned long long __strtol_ull_max_tab[] attribute_hidden;
extern const unsigned char __strtol_ull_rem_tab[] attribute_hidden;
#endif

#define DEF(TYPE, NAME)							   \
  const TYPE NAME[] attribute_hidden =					   \
  {									   \
    F(2), F(3), F(4), F(5), F(6), F(7), F(8), F(9), F(10), 		   \
    F(11), F(12), F(13), F(14), F(15), F(16), F(17), F(18), F(19), F(20),  \
    F(21), F(22), F(23), F(24), F(25), F(26), F(27), F(28), F(29), F(30),  \
    F(31), F(32), F(33), F(34), F(35), F(36)				   \
  }

#if !UNSIGNED && !defined (USE_WIDE_CHAR) && !defined (QUAD)
# define F(X)	ULONG_MAX / X
  DEF (unsigned long, __strtol_ul_max_tab);
# undef F
# define F(X)	ULONG_MAX % X
  DEF (unsigned char, __strtol_ul_rem_tab);
# undef F
#endif
#if !UNSIGNED && !defined (USE_WIDE_CHAR) && defined (QUAD) \
    && __WORDSIZE == 32
# define F(X)	ULONG_LONG_MAX / X
  DEF (unsigned long long, __strtol_ull_max_tab);
# undef F
# define F(X)	ULONG_LONG_MAX % X
  DEF (unsigned char, __strtol_ull_rem_tab);
# undef F
#endif
#undef DEF

/* Define some more readable aliases for these arrays which correspond
   to how they'll be used in the function below.  */
#define jmax_tab	__strtol_ul_max_tab
#if defined(QUAD) && __WORDSIZE == 32
# define cutoff_tab	__strtol_ull_max_tab
# define cutlim_tab	__strtol_ull_rem_tab
#else
# define cutoff_tab	__strtol_ul_max_tab
# define cutlim_tab	__strtol_ul_rem_tab
#endif


/* Convert NPTR to an `unsigned long int' or `long int' in base BASE.
   If BASE is 0 the base is determined by the presence of a leading
   zero, indicating octal or a leading "0x" or "0X", indicating hexadecimal.
   If BASE is < 2 or > 36, it is reset to 10.
   If ENDPTR is not NULL, a pointer to the character after the last
   one converted is stored in *ENDPTR.  */

INT
INTERNAL (__strtol_l) (const STRING_TYPE *nptr, STRING_TYPE **endptr,
		       int base, int group, locale_t loc)
{
  int negative;
  unsigned LONG int cutoff;
  unsigned int cutlim;
  unsigned LONG int i;
  const STRING_TYPE *s;
  UCHAR_TYPE c;
  const STRING_TYPE *save, *end;
  int overflow;
#ifndef USE_WIDE_CHAR
  size_t cnt;
#endif

#ifdef USE_NUMBER_GROUPING
  struct __locale_data *current = loc->__locales[LC_NUMERIC];
  /* The thousands character of the current locale.  */
# ifdef USE_WIDE_CHAR
  wchar_t thousands = L'\0';
# else
  const char *thousands = NULL;
  size_t thousands_len = 0;
# endif
  /* The numeric grouping specification of the current locale,
     in the format described in <locale.h>.  */
  const char *grouping;

  if (__glibc_unlikely (group))
    {
      grouping = _NL_CURRENT (LC_NUMERIC, GROUPING);
      if (*grouping <= 0 || *grouping == CHAR_MAX)
	grouping = NULL;
      else
	{
	  /* Figure out the thousands separator character.  */
# ifdef USE_WIDE_CHAR
#  ifdef _LIBC
	  thousands = _NL_CURRENT_WORD (LC_NUMERIC,
					_NL_NUMERIC_THOUSANDS_SEP_WC);
#  endif
	  if (thousands == L'\0')
	    grouping = NULL;
# else
#  ifdef _LIBC
	  thousands = _NL_CURRENT (LC_NUMERIC, THOUSANDS_SEP);
#  endif
	  if (*thousands == '\0')
	    {
	      thousands = NULL;
	      grouping = NULL;
	    }
# endif
	}
    }
  else
    grouping = NULL;
#endif

  if (base < 0 || base == 1 || base > 36)
    {
      __set_errno (EINVAL);
      return 0;
    }

  save = s = nptr;

  /* Skip white space.  */
  while (ISSPACE (*s))
    ++s;
  if (__glibc_unlikely (*s == L_('\0')))
    goto noconv;

  /* Check for a sign.  */
  negative = 0;
  if (*s == L_('-'))
    {
      negative = 1;
      ++s;
    }
  else if (*s == L_('+'))
    ++s;

  /* Recognize number prefix and if BASE is zero, figure it out ourselves.  */
  if (*s == L_('0'))
    {
      if ((base == 0 || base == 16) && TOUPPER (s[1]) == L_('X'))
	{
	  s += 2;
	  base = 16;
	}
      else if (base == 0)
	base = 8;
    }
  else if (base == 0)
    base = 10;

  /* Save the pointer so we can check later if anything happened.  */
  save = s;

#ifdef USE_NUMBER_GROUPING
  if (base != 10)
    grouping = NULL;

  if (__glibc_unlikely (grouping != NULL))
    {
# ifndef USE_WIDE_CHAR
      thousands_len = strlen (thousands);
# endif

      /* Find the end of the digit string and check its grouping.  */
      end = s;
      if (
# ifdef USE_WIDE_CHAR
	  *s != thousands
# else
	  ({ for (cnt = 0; cnt < thousands_len; ++cnt)
	       if (thousands[cnt] != end[cnt])
		 break;
	     cnt < thousands_len; })
# endif
	  )
	{
	  for (c = *end; c != L_('\0'); c = *++end)
	    if (((STRING_TYPE) c < L_('0') || (STRING_TYPE) c > L_('9'))
# ifdef USE_WIDE_CHAR
		&& (wchar_t) c != thousands
# else
		&& ({ for (cnt = 0; cnt < thousands_len; ++cnt)
			if (thousands[cnt] != end[cnt])
			  break;
		      cnt < thousands_len; })
# endif
		&& (!ISALPHA (c)
		    || (int) (TOUPPER (c) - L_('A') + 10) >= base))
	      break;

# ifdef USE_WIDE_CHAR
	  end = __correctly_grouped_prefixwc (s, end, thousands, grouping);
# else
	  end = __correctly_grouped_prefixmb (s, end, thousands, grouping);
# endif
	}
    }
  else
#endif
    end = NULL;

  /* Avoid runtime division; lookup cutoff and limit.  */
  cutoff = cutoff_tab[base - 2];
  cutlim = cutlim_tab[base - 2];

  overflow = 0;
  i = 0;
  c = *s;
  if (sizeof (long int) != sizeof (LONG int))
    {
      unsigned long int j = 0;
      unsigned long int jmax = jmax_tab[base - 2];

      for (;c != L_('\0'); c = *++s)
	{
	  if (s == end)
	    break;
	  if (c >= L_('0') && c <= L_('9'))
	    c -= L_('0');
#ifdef USE_NUMBER_GROUPING
# ifdef USE_WIDE_CHAR
	  else if (grouping && (wchar_t) c == thousands)
	    continue;
# else
	  else if (thousands_len)
	    {
	      for (cnt = 0; cnt < thousands_len; ++cnt)
		if (thousands[cnt] != s[cnt])
		  break;
	      if (cnt == thousands_len)
		{
		  s += thousands_len - 1;
		  continue;
		}
	      if (ISALPHA (c))
		c = TOUPPER (c) - L_('A') + 10;
	      else
		break;
	    }
# endif
#endif
	  else if (ISALPHA (c))
	    c = TOUPPER (c) - L_('A') + 10;
	  else
	    break;
	  if ((int) c >= base)
	    break;
	  /* Note that we never can have an overflow.  */
	  else if (j >= jmax)
	    {
	      /* We have an overflow.  Now use the long representation.  */
	      i = (unsigned LONG int) j;
	      goto use_long;
	    }
	  else
	    j = j * (unsigned long int) base + c;
	}

      i = (unsigned LONG int) j;
    }
  else
    for (;c != L_('\0'); c = *++s)
      {
	if (s == end)
	  break;
	if (c >= L_('0') && c <= L_('9'))
	  c -= L_('0');
#ifdef USE_NUMBER_GROUPING
# ifdef USE_WIDE_CHAR
	else if (grouping && (wchar_t) c == thousands)
	  continue;
# else
	else if (thousands_len)
	  {
	    for (cnt = 0; cnt < thousands_len; ++cnt)
	      if (thousands[cnt] != s[cnt])
		break;
	    if (cnt == thousands_len)
	      {
		s += thousands_len - 1;
		continue;
	      }
	    if (ISALPHA (c))
	      c = TOUPPER (c) - L_('A') + 10;
	    else
	      break;
	  }
# endif
#endif
	else if (ISALPHA (c))
	  c = TOUPPER (c) - L_('A') + 10;
	else
	  break;
	if ((int) c >= base)
	  break;
	/* Check for overflow.  */
	if (i > cutoff || (i == cutoff && c > cutlim))
	  overflow = 1;
	else
	  {
	  use_long:
	    i *= (unsigned LONG int) base;
	    i += c;
	  }
      }

  /* Check if anything actually happened.  */
  if (s == save)
    goto noconv;

  /* Store in ENDPTR the address of one character
     past the last character we converted.  */
  if (endptr != NULL)
    *endptr = (STRING_TYPE *) s;

#if !UNSIGNED
  /* Check for a value that is within the range of
     `unsigned LONG int', but outside the range of `LONG int'.  */
  if (overflow == 0
      && i > (negative
	      ? -((unsigned LONG int) (STRTOL_LONG_MIN + 1)) + 1
	      : (unsigned LONG int) STRTOL_LONG_MAX))
    overflow = 1;
#endif

  if (__glibc_unlikely (overflow))
    {
      __set_errno (ERANGE);
#if UNSIGNED
      return STRTOL_ULONG_MAX;
#else
      return negative ? STRTOL_LONG_MIN : STRTOL_LONG_MAX;
#endif
    }

  /* Return the result of the appropriate sign.  */
  return negative ? -i : i;

noconv:
  /* We must handle a special case here: the base is 0 or 16 and the
     first two characters are '0' and 'x', but the rest are no
     hexadecimal digits.  This is no error case.  We return 0 and
     ENDPTR points to the `x`.  */
  if (endptr != NULL)
    {
      if (save - nptr >= 2 && TOUPPER (save[-1]) == L_('X')
	  && save[-2] == L_('0'))
	*endptr = (STRING_TYPE *) &save[-1];
      else
	/*  There was no number to convert.  */
	*endptr = (STRING_TYPE *) nptr;
    }

  return 0L;
}
#if defined _LIBC && !defined USE_WIDE_CHAR
libc_hidden_def (INTERNAL (__strtol_l))
#endif

/* External user entry point.  */

#if _LIBC - 0 == 0

/* Prototype.  */
extern INT __strtol_l (const STRING_TYPE *nptr, STRING_TYPE **endptr,
		       int base);
#endif


INT
#ifdef weak_function
weak_function
#endif
__strtol_l (const STRING_TYPE *nptr, STRING_TYPE **endptr,
	    int base, locale_t loc)
{
  return INTERNAL (__strtol_l) (nptr, endptr, base, 0, loc);
}
libc_hidden_def (__strtol_l)
weak_alias (__strtol_l, strtol_l)
