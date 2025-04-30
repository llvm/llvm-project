/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#ifdef _LIBC
# define USE_IN_EXTENDED_LOCALE_MODEL 1
# define HAVE_LIMITS_H 1
# define HAVE_MBLEN 1
# define HAVE_MBRLEN 1
# define HAVE_STRUCT_ERA_ENTRY 1
# define HAVE_TM_GMTOFF 1
# define HAVE_TM_ZONE 1
# define HAVE_TZNAME 1
# define HAVE_TZSET 1
# define HAVE_STRFTIME 0
# define MULTIBYTE_IS_FORMAT_SAFE 1
# define STDC_HEADERS 1
# include "../locale/localeinfo.h"
#endif

#if defined emacs && !defined HAVE_BCOPY
# define HAVE_MEMCPY 1
#endif

#include <ctype.h>
#include <sys/types.h>		/* Some systems define `time_t' here.  */

#ifdef TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# ifdef HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif
#if HAVE_TZNAME
extern char *tzname[];
#endif

/* Do multibyte processing if multibytes are supported, unless
   multibyte sequences are safe in formats.  Multibyte sequences are
   safe if they cannot contain byte sequences that look like format
   conversion specifications.  The GNU C Library uses UTF8 multibyte
   encoding, which is safe for formats, but strftime.c can be used
   with other C libraries that use unsafe encodings.  */
#define DO_MULTIBYTE (HAVE_MBLEN && ! MULTIBYTE_IS_FORMAT_SAFE)

#if DO_MULTIBYTE
# if HAVE_MBRLEN
#  include <wchar.h>
# else
   /* Simulate mbrlen with mblen as best we can.  */
#  define mbstate_t int
#  define mbrlen(s, n, ps) mblen (s, n)
#  define mbsinit(ps) (*(ps) == 0)
# endif
  static const mbstate_t mbstate_zero;
#endif

#if HAVE_LIMITS_H
# include <limits.h>
#endif

#if STDC_HEADERS
# include <stddef.h>
# include <stdlib.h>
# include <string.h>
# include <stdbool.h>
#else
# ifndef HAVE_MEMCPY
#  define memcpy(d, s, n) bcopy ((s), (d), (n))
# endif
#endif

#ifdef COMPILE_WIDE
# include <endian.h>
# define CHAR_T wchar_t
# define UCHAR_T unsigned int
# define L_(Str) L##Str
# define NLW(Sym) _NL_W##Sym

# define MEMCPY(d, s, n) __wmemcpy (d, s, n)
# define STRLEN(s) __wcslen (s)

#else
# define CHAR_T char
# define UCHAR_T unsigned char
# define L_(Str) Str
# define NLW(Sym) Sym
# define ABALTMON_1 _NL_ABALTMON_1

# if !defined STDC_HEADERS && !defined HAVE_MEMCPY
#  define MEMCPY(d, s, n) bcopy ((s), (d), (n))
# else
#  define MEMCPY(d, s, n) memcpy ((d), (s), (n))
# endif
# define STRLEN(s) strlen (s)

# ifdef _LIBC
#  define MEMPCPY(d, s, n) __mempcpy (d, s, n)
# else
#  ifndef HAVE_MEMPCPY
#   define MEMPCPY(d, s, n) ((void *) ((char *) memcpy (d, s, n) + (n)))
#  endif
# endif
#endif

#ifndef PTR
# define PTR void *
#endif

#ifndef CHAR_BIT
# define CHAR_BIT 8
#endif

#ifndef NULL
# define NULL 0
#endif

#define TYPE_SIGNED(t) ((t) -1 < 0)

/* Bound on length of the string representing an integer value of type t.
   Subtract one for the sign bit if t is signed;
   302 / 1000 is log10 (2) rounded up;
   add one for integer division truncation;
   add one more for a minus sign if t is signed.  */
#define INT_STRLEN_BOUND(t) \
 ((sizeof (t) * CHAR_BIT - TYPE_SIGNED (t)) * 302 / 1000 + 1 + TYPE_SIGNED (t))

#define TM_YEAR_BASE 1900

#ifndef __isleap
/* Nonzero if YEAR is a leap year (every 4 years,
   except every 100th isn't, and every 400th is).  */
# define __isleap(year)	\
  ((year) % 4 == 0 && ((year) % 100 != 0 || (year) % 400 == 0))
#endif


#ifdef _LIBC
# define tzname __tzname
# define tzset __tzset
#endif

#if !HAVE_TM_GMTOFF
/* Portable standalone applications should supply a "time_r.h" that
   declares a POSIX-compliant localtime_r, for the benefit of older
   implementations that lack localtime_r or have a nonstandard one.
   Similarly for gmtime_r.  See the gnulib time_r module for one way
   to implement this.  */
# include "time_r.h"
# undef __gmtime_r
# undef __localtime_r
# define __gmtime_r gmtime_r
# define __localtime_r localtime_r
#endif


#if !defined memset && !defined HAVE_MEMSET && !defined _LIBC
/* Some systems lack the `memset' function and we don't want to
   introduce additional dependencies.  */
/* The SGI compiler reportedly barfs on the trailing null
   if we use a string constant as the initializer.  28 June 1997, rms.  */
static const CHAR_T spaces[16] = /* "                " */
{
  L_(' '),L_(' '),L_(' '),L_(' '),L_(' '),L_(' '),L_(' '),L_(' '),
  L_(' '),L_(' '),L_(' '),L_(' '),L_(' '),L_(' '),L_(' '),L_(' ')
};
static const CHAR_T zeroes[16] = /* "0000000000000000" */
{
  L_('0'),L_('0'),L_('0'),L_('0'),L_('0'),L_('0'),L_('0'),L_('0'),
  L_('0'),L_('0'),L_('0'),L_('0'),L_('0'),L_('0'),L_('0'),L_('0')
};

# define memset_space(P, Len) \
  do {									      \
    int _len = (Len);							      \
									      \
    do									      \
      {									      \
	int _this = _len > 16 ? 16 : _len;				      \
	(P) = MEMPCPY ((P), spaces, _this * sizeof (CHAR_T));		      \
	_len -= _this;							      \
      }									      \
    while (_len > 0);							      \
  } while (0)

# define memset_zero(P, Len) \
  do {									      \
    int _len = (Len);							      \
									      \
    do									      \
      {									      \
	int _this = _len > 16 ? 16 : _len;				      \
	(P) = MEMPCPY ((P), zeroes, _this * sizeof (CHAR_T));		      \
	_len -= _this;							      \
      }									      \
    while (_len > 0);							      \
  } while (0)
#else
# ifdef COMPILE_WIDE
#  define memset_space(P, Len) (wmemset ((P), L' ', (Len)), (P) += (Len))
#  define memset_zero(P, Len) (wmemset ((P), L'0', (Len)), (P) += (Len))
# else
#  define memset_space(P, Len) (memset ((P), ' ', (Len)), (P) += (Len))
#  define memset_zero(P, Len) (memset ((P), '0', (Len)), (P) += (Len))
# endif
#endif

#define add(n, f)							      \
  do									      \
    {									      \
      int _n = (n);							      \
      int _delta = width - _n;						      \
      int _incr = _n + (_delta > 0 ? _delta : 0);			      \
      if ((size_t) _incr >= maxsize - i)				      \
	return 0;							      \
      if (p)								      \
	{								      \
	  if (_delta > 0)						      \
	    {								      \
	      if (pad == L_('0'))					      \
		memset_zero (p, _delta);				      \
	      else							      \
		memset_space (p, _delta);				      \
	    }								      \
	  f;								      \
	  p += _n;							      \
	}								      \
      i += _incr;							      \
    } while (0)

#define cpy(n, s) \
    add ((n),								      \
	 if (to_lowcase)						      \
	   memcpy_lowcase (p, (s), _n LOCALE_ARG);			      \
	 else if (to_uppcase)						      \
	   memcpy_uppcase (p, (s), _n LOCALE_ARG);			      \
	 else								      \
	   MEMCPY ((PTR) p, (const PTR) (s), _n))

#ifdef COMPILE_WIDE
# ifndef USE_IN_EXTENDED_LOCALE_MODEL
#  undef __mbsrtowcs_l
#  define __mbsrtowcs_l(d, s, l, st, loc) __mbsrtowcs (d, s, l, st)
# endif
# define widen(os, ws, l) \
  {									      \
    mbstate_t __st;							      \
    const char *__s = os;						      \
    memset (&__st, '\0', sizeof (__st));				      \
    l = __mbsrtowcs_l (NULL, &__s, 0, &__st, loc);			      \
    ws = alloca ((l + 1) * sizeof (wchar_t));				      \
    (void) __mbsrtowcs_l (ws, &__s, l, &__st, loc);			      \
  }
#endif


#if defined _LIBC && defined USE_IN_EXTENDED_LOCALE_MODEL
/* We use this code also for the extended locale handling where the
   function gets as an additional argument the locale which has to be
   used.  To access the values we have to redefine the _NL_CURRENT
   macro.  */
# define strftime		__strftime_l
# define wcsftime		__wcsftime_l
# undef _NL_CURRENT
# define _NL_CURRENT(category, item) \
  (current->values[_NL_ITEM_INDEX (item)].string)
# define LOCALE_PARAM , locale_t loc
# define LOCALE_ARG , loc
# define HELPER_LOCALE_ARG  , current
#else
# define LOCALE_PARAM
# define LOCALE_ARG
# ifdef _LIBC
#  define HELPER_LOCALE_ARG , _NL_CURRENT_DATA (LC_TIME)
# else
#  define HELPER_LOCALE_ARG
# endif
#endif

#ifdef COMPILE_WIDE
# ifdef USE_IN_EXTENDED_LOCALE_MODEL
#  define TOUPPER(Ch, L) __towupper_l (Ch, L)
#  define TOLOWER(Ch, L) __towlower_l (Ch, L)
# else
#  define TOUPPER(Ch, L) towupper (Ch)
#  define TOLOWER(Ch, L) towlower (Ch)
# endif
#else
# ifdef _LIBC
#  ifdef USE_IN_EXTENDED_LOCALE_MODEL
#   define TOUPPER(Ch, L) __toupper_l (Ch, L)
#   define TOLOWER(Ch, L) __tolower_l (Ch, L)
#  else
#   define TOUPPER(Ch, L) toupper (Ch)
#   define TOLOWER(Ch, L) tolower (Ch)
#  endif
# else
#  define TOUPPER(Ch, L) (islower (Ch) ? toupper (Ch) : (Ch))
#  define TOLOWER(Ch, L) (isupper (Ch) ? tolower (Ch) : (Ch))
# endif
#endif
/* We don't use `isdigit' here since the locale dependent
   interpretation is not what we want here.  We only need to accept
   the arabic digits in the ASCII range.  One day there is perhaps a
   more reliable way to accept other sets of digits.  */
#define ISDIGIT(Ch) ((unsigned int) (Ch) - L_('0') <= 9)

static CHAR_T *memcpy_lowcase (CHAR_T *dest, const CHAR_T *src,
			       size_t len LOCALE_PARAM) __THROW;

static CHAR_T *
memcpy_lowcase (CHAR_T *dest, const CHAR_T *src, size_t len LOCALE_PARAM)
{
  while (len-- > 0)
    dest[len] = TOLOWER ((UCHAR_T) src[len], loc);
  return dest;
}

static CHAR_T *memcpy_uppcase (CHAR_T *dest, const CHAR_T *src,
			       size_t len LOCALE_PARAM) __THROW;

static CHAR_T *
memcpy_uppcase (CHAR_T *dest, const CHAR_T *src, size_t len LOCALE_PARAM)
{
  while (len-- > 0)
    dest[len] = TOUPPER ((UCHAR_T) src[len], loc);
  return dest;
}


#if ! HAVE_TM_GMTOFF
/* Yield the difference between *A and *B,
   measured in seconds, ignoring leap seconds.  */
# define tm_diff ftime_tm_diff
static int tm_diff (const struct tm *, const struct tm *) __THROW;
static int
tm_diff (const struct tm *a, const struct tm *b)
{
  /* Compute intervening leap days correctly even if year is negative.
     Take care to avoid int overflow in leap day calculations,
     but it's OK to assume that A and B are close to each other.  */
  int a4 = (a->tm_year >> 2) + (TM_YEAR_BASE >> 2) - ! (a->tm_year & 3);
  int b4 = (b->tm_year >> 2) + (TM_YEAR_BASE >> 2) - ! (b->tm_year & 3);
  int a100 = a4 / 25 - (a4 % 25 < 0);
  int b100 = b4 / 25 - (b4 % 25 < 0);
  int a400 = a100 >> 2;
  int b400 = b100 >> 2;
  int intervening_leap_days = (a4 - b4) - (a100 - b100) + (a400 - b400);
  int years = a->tm_year - b->tm_year;
  int days = (365 * years + intervening_leap_days
	      + (a->tm_yday - b->tm_yday));
  return (60 * (60 * (24 * days + (a->tm_hour - b->tm_hour))
		+ (a->tm_min - b->tm_min))
	  + (a->tm_sec - b->tm_sec));
}
#endif /* ! HAVE_TM_GMTOFF */



/* The number of days from the first day of the first ISO week of this
   year to the year day YDAY with week day WDAY.  ISO weeks start on
   Monday; the first ISO week has the year's first Thursday.  YDAY may
   be as small as YDAY_MINIMUM.  */
#define ISO_WEEK_START_WDAY 1 /* Monday */
#define ISO_WEEK1_WDAY 4 /* Thursday */
#define YDAY_MINIMUM (-366)
static int iso_week_days (int, int) __THROW;
#ifdef __GNUC__
__inline__
#endif
static int
iso_week_days (int yday, int wday)
{
  /* Add enough to the first operand of % to make it nonnegative.  */
  int big_enough_multiple_of_7 = (-YDAY_MINIMUM / 7 + 2) * 7;
  return (yday
	  - (yday - wday + ISO_WEEK1_WDAY + big_enough_multiple_of_7) % 7
	  + ISO_WEEK1_WDAY - ISO_WEEK_START_WDAY);
}


#if !(defined _NL_CURRENT || HAVE_STRFTIME)
static CHAR_T const weekday_name[][10] =
  {
    L_("Sunday"), L_("Monday"), L_("Tuesday"), L_("Wednesday"),
    L_("Thursday"), L_("Friday"), L_("Saturday")
  };
static CHAR_T const month_name[][10] =
  {
    L_("January"), L_("February"), L_("March"), L_("April"), L_("May"),
    L_("June"), L_("July"), L_("August"), L_("September"), L_("October"),
    L_("November"), L_("December")
  };
#endif


#ifdef emacs
# define my_strftime emacs_strftimeu
# define ut_argument , ut
# define ut_argument_spec , int ut
#else
# ifdef COMPILE_WIDE
#  define my_strftime wcsftime
#  define nl_get_alt_digit _nl_get_walt_digit
# else
#  define my_strftime strftime
#  define nl_get_alt_digit _nl_get_alt_digit
# endif
# define ut_argument
# define ut_argument_spec
/* We don't have this information in general.  */
# define ut 0
#endif

static size_t __strftime_internal (CHAR_T *, size_t, const CHAR_T *,
				   const struct tm *, int, bool *
				   ut_argument_spec
				   LOCALE_PARAM) __THROW;

/* Write information from TP into S according to the format
   string FORMAT, writing no more that MAXSIZE characters
   (including the terminating '\0') and returning number of
   characters written.  If S is NULL, nothing will be written
   anywhere, so to determine how many characters would be
   written, use NULL for S and (size_t) UINT_MAX for MAXSIZE.  */

size_t
my_strftime (CHAR_T *s, size_t maxsize, const CHAR_T *format,
	     const struct tm *tp ut_argument_spec LOCALE_PARAM)
{
#if !defined _LIBC && HAVE_TZNAME && HAVE_TZSET
  /* Solaris 2.5 tzset sometimes modifies the storage returned by localtime.
     Work around this bug by copying *tp before it might be munged.  */
  struct tm tmcopy;
  tmcopy = *tp;
  tp = &tmcopy;
#endif
  bool tzset_called = false;
  return __strftime_internal (s, maxsize, format, tp, 0, &tzset_called
			      ut_argument LOCALE_ARG);
}
#ifdef _LIBC
libc_hidden_def (my_strftime)
#endif

static size_t
__strftime_internal (CHAR_T *s, size_t maxsize, const CHAR_T *format,
		     const struct tm *tp, int yr_spec, bool *tzset_called
		     ut_argument_spec LOCALE_PARAM)
{
#if defined _LIBC && defined USE_IN_EXTENDED_LOCALE_MODEL
  struct __locale_data *const current = loc->__locales[LC_TIME];
#endif

  int hour12 = tp->tm_hour;
#ifdef _NL_CURRENT
  /* We cannot make the following values variables since we must delay
     the evaluation of these values until really needed since some
     expressions might not be valid in every situation.  The `struct tm'
     might be generated by a strptime() call that initialized
     only a few elements.  Dereference the pointers only if the format
     requires this.  Then it is ok to fail if the pointers are invalid.  */
# define a_wkday \
  ((const CHAR_T *) (tp->tm_wday < 0 || tp->tm_wday > 6			     \
		     ? "?" : _NL_CURRENT (LC_TIME, NLW(ABDAY_1) + tp->tm_wday)))
# define f_wkday \
  ((const CHAR_T *) (tp->tm_wday < 0 || tp->tm_wday > 6			     \
		     ? "?" : _NL_CURRENT (LC_TIME, NLW(DAY_1) + tp->tm_wday)))
# define a_month \
  ((const CHAR_T *) (tp->tm_mon < 0 || tp->tm_mon > 11			     \
		     ? "?" : _NL_CURRENT (LC_TIME, NLW(ABMON_1) + tp->tm_mon)))
# define f_month \
  ((const CHAR_T *) (tp->tm_mon < 0 || tp->tm_mon > 11			     \
		     ? "?" : _NL_CURRENT (LC_TIME, NLW(MON_1) + tp->tm_mon)))
# define a_altmonth \
  ((const CHAR_T *) (tp->tm_mon < 0 || tp->tm_mon > 11			     \
		     ? "?" : _NL_CURRENT (LC_TIME, NLW(ABALTMON_1) + tp->tm_mon)))
# define f_altmonth \
  ((const CHAR_T *) (tp->tm_mon < 0 || tp->tm_mon > 11			     \
		     ? "?" : _NL_CURRENT (LC_TIME, NLW(ALTMON_1) + tp->tm_mon)))
# define ampm \
  ((const CHAR_T *) _NL_CURRENT (LC_TIME, tp->tm_hour > 11		      \
				 ? NLW(PM_STR) : NLW(AM_STR)))

# define aw_len STRLEN (a_wkday)
# define am_len STRLEN (a_month)
# define aam_len STRLEN (a_altmonth)
# define ap_len STRLEN (ampm)
#else
# if !HAVE_STRFTIME
#  define f_wkday (tp->tm_wday < 0 || tp->tm_wday > 6	\
		   ? "?" : weekday_name[tp->tm_wday])
#  define f_month (tp->tm_mon < 0 || tp->tm_mon > 11	\
		   ? "?" : month_name[tp->tm_mon])
#  define a_wkday f_wkday
#  define a_month f_month
#  define a_altmonth a_month
#  define f_altmonth f_month
#  define ampm (L_("AMPM") + 2 * (tp->tm_hour > 11))

  size_t aw_len = 3;
  size_t am_len = 3;
  size_t aam_len = 3;
  size_t ap_len = 2;
# endif
#endif
  const char *zone;
  size_t i = 0;
  CHAR_T *p = s;
  const CHAR_T *f;
#if DO_MULTIBYTE && !defined COMPILE_WIDE
  const char *format_end = NULL;
#endif

  zone = NULL;
#if HAVE_TM_ZONE
  /* The POSIX test suite assumes that setting
     the environment variable TZ to a new value before calling strftime()
     will influence the result (the %Z format) even if the information in
     TP is computed with a totally different time zone.
     This is bogus: though POSIX allows bad behavior like this,
     POSIX does not require it.  Do the right thing instead.  */
  zone = (const char *) tp->tm_zone;
#endif
#if HAVE_TZNAME
  if (ut)
    {
      if (! (zone && *zone))
	zone = "GMT";
    }
#endif

  if (hour12 > 12)
    hour12 -= 12;
  else
    if (hour12 == 0)
      hour12 = 12;

  for (f = format; *f != '\0'; ++f)
    {
      int pad = 0;		/* Padding for number ('-', '_', or 0).  */
      int modifier;		/* Field modifier ('E', 'O', or 0).  */
      int digits;		/* Max digits for numeric format.  */
      int number_value;		/* Numeric value to be printed.  */
      int negative_number;	/* 1 if the number is negative.  */
      const CHAR_T *subfmt;
      CHAR_T *bufp;
      CHAR_T buf[1 + (sizeof (int) < sizeof (time_t)
		      ? INT_STRLEN_BOUND (time_t)
		      : INT_STRLEN_BOUND (int))];
      int width = -1;
      int to_lowcase = 0;
      int to_uppcase = 0;
      int change_case = 0;
      int format_char;

#if DO_MULTIBYTE && !defined COMPILE_WIDE
      switch (*f)
	{
	case L_('%'):
	  break;

	case L_('\b'): case L_('\t'): case L_('\n'):
	case L_('\v'): case L_('\f'): case L_('\r'):
	case L_(' '): case L_('!'): case L_('"'): case L_('#'): case L_('&'):
	case L_('\''): case L_('('): case L_(')'): case L_('*'): case L_('+'):
	case L_(','): case L_('-'): case L_('.'): case L_('/'): case L_('0'):
	case L_('1'): case L_('2'): case L_('3'): case L_('4'): case L_('5'):
	case L_('6'): case L_('7'): case L_('8'): case L_('9'): case L_(':'):
	case L_(';'): case L_('<'): case L_('='): case L_('>'): case L_('?'):
	case L_('A'): case L_('B'): case L_('C'): case L_('D'): case L_('E'):
	case L_('F'): case L_('G'): case L_('H'): case L_('I'): case L_('J'):
	case L_('K'): case L_('L'): case L_('M'): case L_('N'): case L_('O'):
	case L_('P'): case L_('Q'): case L_('R'): case L_('S'): case L_('T'):
	case L_('U'): case L_('V'): case L_('W'): case L_('X'): case L_('Y'):
	case L_('Z'): case L_('['): case L_('\\'): case L_(']'): case L_('^'):
	case L_('_'): case L_('a'): case L_('b'): case L_('c'): case L_('d'):
	case L_('e'): case L_('f'): case L_('g'): case L_('h'): case L_('i'):
	case L_('j'): case L_('k'): case L_('l'): case L_('m'): case L_('n'):
	case L_('o'): case L_('p'): case L_('q'): case L_('r'): case L_('s'):
	case L_('t'): case L_('u'): case L_('v'): case L_('w'): case L_('x'):
	case L_('y'): case L_('z'): case L_('{'): case L_('|'): case L_('}'):
	case L_('~'):
	  /* The C Standard requires these 98 characters (plus '%') to
	     be in the basic execution character set.  None of these
	     characters can start a multibyte sequence, so they need
	     not be analyzed further.  */
	  add (1, *p = *f);
	  continue;

	default:
	  /* Copy this multibyte sequence until we reach its end, find
	     an error, or come back to the initial shift state.  */
	  {
	    mbstate_t mbstate = mbstate_zero;
	    size_t len = 0;
	    size_t fsize;

	    if (! format_end)
	      format_end = f + strlen (f) + 1;
	    fsize = format_end - f;

	    do
	      {
		size_t bytes = mbrlen (f + len, fsize - len, &mbstate);

		if (bytes == 0)
		  break;

		if (bytes == (size_t) -2)
		  {
		    len += strlen (f + len);
		    break;
		  }

		if (bytes == (size_t) -1)
		  {
		    len++;
		    break;
		  }

		len += bytes;
	      }
	    while (! mbsinit (&mbstate));

	    cpy (len, f);
	    f += len - 1;
	    continue;
	  }
	}

#else /* ! DO_MULTIBYTE */

      /* Either multibyte encodings are not supported, they are
	 safe for formats, so any non-'%' byte can be copied through,
	 or this is the wide character version.  */
      if (*f != L_('%'))
	{
	  add (1, *p = *f);
	  continue;
	}

#endif /* ! DO_MULTIBYTE */

      /* Check for flags that can modify a format.  */
      while (1)
	{
	  switch (*++f)
	    {
	      /* This influences the number formats.  */
	    case L_('_'):
	    case L_('-'):
	    case L_('0'):
	      pad = *f;
	      continue;

	      /* This changes textual output.  */
	    case L_('^'):
	      to_uppcase = 1;
	      continue;
	    case L_('#'):
	      change_case = 1;
	      continue;

	    default:
	      break;
	    }
	  break;
	}

      /* As a GNU extension we allow to specify the field width.  */
      if (ISDIGIT (*f))
	{
	  width = 0;
	  do
	    {
	      if (width > INT_MAX / 10
		  || (width == INT_MAX / 10 && *f - L_('0') > INT_MAX % 10))
		/* Avoid overflow.  */
		width = INT_MAX;
	      else
		{
		  width *= 10;
		  width += *f - L_('0');
		}
	      ++f;
	    }
	  while (ISDIGIT (*f));
	}

      /* Check for modifiers.  */
      switch (*f)
	{
	case L_('E'):
	case L_('O'):
	  modifier = *f++;
	  break;

	default:
	  modifier = 0;
	  break;
	}

      /* Now do the specified format.  */
      format_char = *f;
      switch (format_char)
	{
#define DO_NUMBER(d, v)				\
	  do					\
	    {					\
	      digits = d > width ? d : width;	\
	      number_value = v;			\
	      goto do_number;			\
	    }					\
	  while (0)
#define DO_NUMBER_SPACEPAD(d, v)		\
	  do					\
	    {					\
	      digits = d > width ? d : width;	\
	      number_value = v;			\
	      goto do_number_spacepad;		\
	    }					\
	  while (0)

	case L_('%'):
	  if (modifier != 0)
	    goto bad_format;
	  add (1, *p = *f);
	  break;

	case L_('a'):
	  if (modifier != 0)
	    goto bad_format;
	  if (change_case)
	    {
	      to_uppcase = 1;
	      to_lowcase = 0;
	    }
#if defined _NL_CURRENT || !HAVE_STRFTIME
	  cpy (aw_len, a_wkday);
	  break;
#else
	  goto underlying_strftime;
#endif

	case 'A':
	  if (modifier != 0)
	    goto bad_format;
	  if (change_case)
	    {
	      to_uppcase = 1;
	      to_lowcase = 0;
	    }
#if defined _NL_CURRENT || !HAVE_STRFTIME
	  cpy (STRLEN (f_wkday), f_wkday);
	  break;
#else
	  goto underlying_strftime;
#endif

	case L_('b'):
	case L_('h'):
	  if (change_case)
	    {
	      to_uppcase = 1;
	      to_lowcase = 0;
	    }
	  if (modifier == L_('E'))
	    goto bad_format;
#if defined _NL_CURRENT || !HAVE_STRFTIME
	  if (modifier == L_('O'))
	    cpy (aam_len, a_altmonth);
	  else
	    cpy (am_len, a_month);
	  break;
#else
	  goto underlying_strftime;
#endif

	case L_('B'):
	  if (modifier == L_('E'))
	    goto bad_format;
	  if (change_case)
	    {
	      to_uppcase = 1;
	      to_lowcase = 0;
	    }
#if defined _NL_CURRENT || !HAVE_STRFTIME
	  if (modifier == L_('O'))
	    cpy (STRLEN (f_altmonth), f_altmonth);
	  else
	    cpy (STRLEN (f_month), f_month);
	  break;
#else
	  goto underlying_strftime;
#endif

	case L_('c'):
	  if (modifier == L_('O'))
	    goto bad_format;
#ifdef _NL_CURRENT
	  if (! (modifier == L_('E')
		 && (*(subfmt =
		       (const CHAR_T *) _NL_CURRENT (LC_TIME,
						     NLW(ERA_D_T_FMT)))
		     != '\0')))
	    subfmt = (const CHAR_T *) _NL_CURRENT (LC_TIME, NLW(D_T_FMT));
#else
# if HAVE_STRFTIME
	  goto underlying_strftime;
# else
	  subfmt = L_("%a %b %e %H:%M:%S %Y");
# endif
#endif

	subformat:
	  {
	    CHAR_T *old_start = p;
	    size_t len = __strftime_internal (NULL, (size_t) -1, subfmt,
					      tp, yr_spec, tzset_called
					      ut_argument LOCALE_ARG);
	    add (len, __strftime_internal (p, maxsize - i, subfmt,
					   tp, yr_spec, tzset_called
					   ut_argument LOCALE_ARG));

	    if (to_uppcase)
	      while (old_start < p)
		{
		  *old_start = TOUPPER ((UCHAR_T) *old_start, loc);
		  ++old_start;
		}
	  }
	  break;

#if HAVE_STRFTIME && ! (defined _NL_CURRENT && HAVE_STRUCT_ERA_ENTRY)
	underlying_strftime:
	  {
	    /* The relevant information is available only via the
	       underlying strftime implementation, so use that.  */
	    char ufmt[4];
	    char *u = ufmt;
	    char ubuf[1024]; /* enough for any single format in practice */
	    size_t len;
	    /* Make sure we're calling the actual underlying strftime.
	       In some cases, config.h contains something like
	       "#define strftime rpl_strftime".  */
# ifdef strftime
#  undef strftime
	    size_t strftime ();
# endif

	    *u++ = '%';
	    if (modifier != 0)
	      *u++ = modifier;
	    *u++ = format_char;
	    *u = '\0';
	    len = strftime (ubuf, sizeof ubuf, ufmt, tp);
	    if (len == 0 && ubuf[0] != '\0')
	      return 0;
	    cpy (len, ubuf);
	  }
	  break;
#endif

	case L_('C'):
	  if (modifier == L_('E'))
	    {
#if HAVE_STRUCT_ERA_ENTRY
	      struct era_entry *era = _nl_get_era_entry (tp HELPER_LOCALE_ARG);
	      if (era)
		{
# ifdef COMPILE_WIDE
		  size_t len = __wcslen (era->era_wname);
		  cpy (len, era->era_wname);
# else
		  size_t len = strlen (era->era_name);
		  cpy (len, era->era_name);
# endif
		  break;
		}
#else
# if HAVE_STRFTIME
	      goto underlying_strftime;
# endif
#endif
	    }

	  {
	    int year = tp->tm_year + TM_YEAR_BASE;
	    DO_NUMBER (1, year / 100 - (year % 100 < 0));
	  }

	case L_('x'):
	  if (modifier == L_('O'))
	    goto bad_format;
#ifdef _NL_CURRENT
	  if (! (modifier == L_('E')
		 && (*(subfmt =
		       (const CHAR_T *) _NL_CURRENT (LC_TIME, NLW(ERA_D_FMT)))
		     != L_('\0'))))
	    subfmt = (const CHAR_T *) _NL_CURRENT (LC_TIME, NLW(D_FMT));
	  goto subformat;
#else
# if HAVE_STRFTIME
	  goto underlying_strftime;
# else
	  /* Fall through.  */
# endif
#endif
	case L_('D'):
	  if (modifier != 0)
	    goto bad_format;
	  subfmt = L_("%m/%d/%y");
	  goto subformat;

	case L_('d'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (2, tp->tm_mday);

	case L_('e'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER_SPACEPAD (2, tp->tm_mday);

	  /* All numeric formats set DIGITS and NUMBER_VALUE and then
	     jump to one of these two labels.  */

	do_number_spacepad:
	  /* Force `_' flag unless overwritten by `0' or '-' flag.  */
	  if (pad != L_('0') && pad != L_('-'))
	    pad = L_('_');

	do_number:
	  /* Format the number according to the MODIFIER flag.  */

	  if (modifier == L_('O') && 0 <= number_value)
	    {
#ifdef _NL_CURRENT
	      /* Get the locale specific alternate representation of
		 the number NUMBER_VALUE.  If none exist NULL is returned.  */
	      const CHAR_T *cp = nl_get_alt_digit (number_value
						   HELPER_LOCALE_ARG);

	      if (cp != NULL)
		{
		  size_t digitlen = STRLEN (cp);
		  if (digitlen != 0)
		    {
		      cpy (digitlen, cp);
		      break;
		    }
		}
#else
# if HAVE_STRFTIME
	      goto underlying_strftime;
# endif
#endif
	    }
	  {
	    unsigned int u = number_value;

	    bufp = buf + sizeof (buf) / sizeof (buf[0]);
	    negative_number = number_value < 0;

	    if (negative_number)
	      u = -u;

	    do
	      *--bufp = u % 10 + L_('0');
	    while ((u /= 10) != 0);
	  }

	do_number_sign_and_padding:
	  if (negative_number)
	    *--bufp = L_('-');

	  if (pad != L_('-'))
	    {
	      int padding = digits - (buf + (sizeof (buf) / sizeof (buf[0]))
				      - bufp);

	      if (padding > 0)
		{
		  if (pad == L_('_'))
		    {
		      if ((size_t) padding >= maxsize - i)
			return 0;

		      if (p)
			memset_space (p, padding);
		      i += padding;
		      width = width > padding ? width - padding : 0;
		    }
		  else
		    {
		      if ((size_t) digits >= maxsize - i)
			return 0;

		      if (negative_number)
			{
			  ++bufp;

			  if (p)
			    *p++ = L_('-');
			  ++i;
			}

		      if (p)
			memset_zero (p, padding);
		      i += padding;
		      width = 0;
		    }
		}
	    }

	  cpy (buf + sizeof (buf) / sizeof (buf[0]) - bufp, bufp);
	  break;

	case L_('F'):
	  if (modifier != 0)
	    goto bad_format;
	  subfmt = L_("%Y-%m-%d");
	  goto subformat;

	case L_('H'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (2, tp->tm_hour);

	case L_('I'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (2, hour12);

	case L_('k'):		/* GNU extension.  */
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER_SPACEPAD (2, tp->tm_hour);

	case L_('l'):		/* GNU extension.  */
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER_SPACEPAD (2, hour12);

	case L_('j'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (3, 1 + tp->tm_yday);

	case L_('M'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (2, tp->tm_min);

	case L_('m'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (2, tp->tm_mon + 1);

	case L_('n'):
	  add (1, *p = L_('\n'));
	  break;

	case L_('P'):
	  to_lowcase = 1;
#if !defined _NL_CURRENT && HAVE_STRFTIME
	  format_char = L_('p');
#endif
	  /* FALLTHROUGH */

	case L_('p'):
	  if (change_case)
	    {
	      to_uppcase = 0;
	      to_lowcase = 1;
	    }
#if defined _NL_CURRENT || !HAVE_STRFTIME
	  cpy (ap_len, ampm);
	  break;
#else
	  goto underlying_strftime;
#endif

	case L_('R'):
	  subfmt = L_("%H:%M");
	  goto subformat;

	case L_('r'):
#if !defined _NL_CURRENT && HAVE_STRFTIME
	  goto underlying_strftime;
#else
# ifdef _NL_CURRENT
	  if (*(subfmt = (const CHAR_T *) _NL_CURRENT (LC_TIME,
						       NLW(T_FMT_AMPM)))
	      == L_('\0'))
# endif
	    subfmt = L_("%I:%M:%S %p");
	  goto subformat;
#endif

	case L_('S'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (2, tp->tm_sec);

	case L_('s'):		/* GNU extension.  */
	  {
	    struct tm ltm;
	    time_t t;

	    ltm = *tp;
	    t = mktime (&ltm);

	    /* Generate string value for T using time_t arithmetic;
	       this works even if sizeof (long) < sizeof (time_t).  */

	    bufp = buf + sizeof (buf) / sizeof (buf[0]);
	    negative_number = t < 0;

	    do
	      {
		int d = t % 10;
		t /= 10;

		if (negative_number)
		  {
		    d = -d;

		    /* Adjust if division truncates to minus infinity.  */
		    if (0 < -1 % 10 && d < 0)
		      {
			t++;
			d += 10;
		      }
		  }

		*--bufp = d + L_('0');
	      }
	    while (t != 0);

	    digits = 1;
	    goto do_number_sign_and_padding;
	  }

	case L_('X'):
	  if (modifier == L_('O'))
	    goto bad_format;
#ifdef _NL_CURRENT
	  if (! (modifier == L_('E')
		 && (*(subfmt =
		       (const CHAR_T *) _NL_CURRENT (LC_TIME, NLW(ERA_T_FMT)))
		     != L_('\0'))))
	    subfmt = (const CHAR_T *) _NL_CURRENT (LC_TIME, NLW(T_FMT));
	  goto subformat;
#else
# if HAVE_STRFTIME
	  goto underlying_strftime;
# else
	  /* Fall through.  */
# endif
#endif
	case L_('T'):
	  subfmt = L_("%H:%M:%S");
	  goto subformat;

	case L_('t'):
	  add (1, *p = L_('\t'));
	  break;

	case L_('u'):
	  DO_NUMBER (1, (tp->tm_wday - 1 + 7) % 7 + 1);

	case L_('U'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (2, (tp->tm_yday - tp->tm_wday + 7) / 7);

	case L_('V'):
	case L_('g'):
	case L_('G'):
	  if (modifier == L_('E'))
	    goto bad_format;
	  {
	    int year = tp->tm_year + TM_YEAR_BASE;
	    int days = iso_week_days (tp->tm_yday, tp->tm_wday);

	    if (days < 0)
	      {
		/* This ISO week belongs to the previous year.  */
		year--;
		days = iso_week_days (tp->tm_yday + (365 + __isleap (year)),
				      tp->tm_wday);
	      }
	    else
	      {
		int d = iso_week_days (tp->tm_yday - (365 + __isleap (year)),
				       tp->tm_wday);
		if (0 <= d)
		  {
		    /* This ISO week belongs to the next year.  */
		    year++;
		    days = d;
		  }
	      }

	    switch (*f)
	      {
	      case L_('g'):
		DO_NUMBER (2, (year % 100 + 100) % 100);

	      case L_('G'):
		DO_NUMBER (1, year);

	      default:
		DO_NUMBER (2, days / 7 + 1);
	      }
	  }

	case L_('W'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (2, (tp->tm_yday - (tp->tm_wday - 1 + 7) % 7 + 7) / 7);

	case L_('w'):
	  if (modifier == L_('E'))
	    goto bad_format;

	  DO_NUMBER (1, tp->tm_wday);

	case L_('Y'):
	  if (modifier == L_('E'))
	    {
#if HAVE_STRUCT_ERA_ENTRY
	      struct era_entry *era = _nl_get_era_entry (tp HELPER_LOCALE_ARG);
	      if (era)
		{
# ifdef COMPILE_WIDE
		  subfmt = era->era_wformat;
# else
		  subfmt = era->era_format;
# endif
		  if (pad != 0)
		    yr_spec = pad;
		  goto subformat;
		}
#else
# if HAVE_STRFTIME
	      goto underlying_strftime;
# endif
#endif
	    }
	  if (modifier == L_('O'))
	    goto bad_format;
	  else
	    DO_NUMBER (1, tp->tm_year + TM_YEAR_BASE);

	case L_('y'):
	  if (modifier == L_('E'))
	    {
#if HAVE_STRUCT_ERA_ENTRY
	      struct era_entry *era = _nl_get_era_entry (tp HELPER_LOCALE_ARG);
	      if (era)
		{
		  int delta = tp->tm_year - era->start_date[0];
		  if (yr_spec != 0)
		    pad = yr_spec;
		  DO_NUMBER (2, (era->offset
				 + delta * era->absolute_direction));
		}
#else
# if HAVE_STRFTIME
	      goto underlying_strftime;
# endif
#endif
	    }
	  DO_NUMBER (2, (tp->tm_year % 100 + 100) % 100);

	case L_('Z'):
	  if (change_case)
	    {
	      to_uppcase = 0;
	      to_lowcase = 1;
	    }

#if HAVE_TZNAME
	  /* The tzset() call might have changed the value.  */
	  if (!(zone && *zone) && tp->tm_isdst >= 0)
	    {
	      /* POSIX.1 requires that local time zone information is used as
		 though strftime called tzset.  */
# if HAVE_TZSET
	      if (!*tzset_called)
		{
		  tzset ();
		  *tzset_called = true;
		}
# endif
	      zone = tp->tm_isdst <= 1 ? tzname[tp->tm_isdst] : "?";
	    }
#endif
	  if (! zone)
	    zone = "";

#ifdef COMPILE_WIDE
	  {
	    /* The zone string is always given in multibyte form.  We have
	       to transform it first.  */
	    wchar_t *wczone;
	    size_t len;
	    widen (zone, wczone, len);
	    cpy (len, wczone);
	  }
#else
	  cpy (strlen (zone), zone);
#endif
	  break;

	case L_('z'):
	  if (tp->tm_isdst < 0)
	    break;

	  {
	    int diff;
#if HAVE_TM_GMTOFF
	    diff = tp->tm_gmtoff;
#else
	    if (ut)
	      diff = 0;
	    else
	      {
		struct tm gtm;
		struct tm ltm;
		time_t lt;

		/* POSIX.1 requires that local time zone information is used as
		   though strftime called tzset.  */
# if HAVE_TZSET
		if (!*tzset_called)
		  {
		    tzset ();
		    *tzset_called = true;
		  }
# endif

		ltm = *tp;
		lt = mktime (&ltm);

		if (lt == (time_t) -1)
		  {
		    /* mktime returns -1 for errors, but -1 is also a
		       valid time_t value.  Check whether an error really
		       occurred.  */
		    struct tm tm;

		    if (! __localtime_r (&lt, &tm)
			|| ((ltm.tm_sec ^ tm.tm_sec)
			    | (ltm.tm_min ^ tm.tm_min)
			    | (ltm.tm_hour ^ tm.tm_hour)
			    | (ltm.tm_mday ^ tm.tm_mday)
			    | (ltm.tm_mon ^ tm.tm_mon)
			    | (ltm.tm_year ^ tm.tm_year)))
		      break;
		  }

		if (! __gmtime_r (&lt, &gtm))
		  break;

		diff = tm_diff (&ltm, &gtm);
	      }
#endif

	    if (diff < 0)
	      {
		add (1, *p = L_('-'));
		diff = -diff;
	      }
	    else
	      add (1, *p = L_('+'));

	    diff /= 60;
	    DO_NUMBER (4, (diff / 60) * 100 + diff % 60);
	  }

	case L_('\0'):		/* GNU extension: % at end of format.  */
	    --f;
	    /* Fall through.  */
	default:
	  /* Unknown format; output the format, including the '%',
	     since this is most likely the right thing to do if a
	     multibyte string has been misparsed.  */
	bad_format:
	  {
	    int flen;
	    for (flen = 1; f[1 - flen] != L_('%'); flen++)
	      continue;
	    cpy (flen, &f[1 - flen]);
	  }
	  break;
	}
    }

  if (p && maxsize != 0)
    *p = L_('\0');
  return i;
}


#ifdef emacs
/* For Emacs we have a separate interface which corresponds to the normal
   strftime function and does not have the extra information whether the
   TP arguments comes from a `gmtime' call or not.  */
size_t
emacs_strftime (char *s, size_t maxsize, const char *format,
		const struct tm *tp)
{
  return my_strftime (s, maxsize, format, tp, 0);
}
#endif

#if defined _LIBC && !defined COMPILE_WIDE
weak_alias (__strftime_l, strftime_l)
#endif
