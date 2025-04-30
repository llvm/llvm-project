/* Convert a 'struct tm' to a time_t value.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Paul Eggert <eggert@twinsun.com>.

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

/* The following macros influence what gets defined when this file is compiled:

   Macro/expression            Which gnulib module    This compilation unit
                                                      should define

   _LIBC                       (glibc proper)         mktime

   NEED_MKTIME_WORKING         mktime                 rpl_mktime
   || NEED_MKTIME_WINDOWS

   NEED_MKTIME_INTERNAL        mktime-internal        mktime_internal
 */

#ifndef _LIBC
# include <libc-config.h>
#endif

/* Assume that leap seconds are possible, unless told otherwise.
   If the host has a 'zic' command with a '-L leapsecondfilename' option,
   then it supports leap seconds; otherwise it probably doesn't.  */
#ifndef LEAP_SECONDS_POSSIBLE
# define LEAP_SECONDS_POSSIBLE 1
#endif

#include <time.h>

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <intprops.h>
#include <verify.h>

#ifndef NEED_MKTIME_INTERNAL
# define NEED_MKTIME_INTERNAL 0
#endif
#ifndef NEED_MKTIME_WINDOWS
# define NEED_MKTIME_WINDOWS 0
#endif
#ifndef NEED_MKTIME_WORKING
# define NEED_MKTIME_WORKING 0
#endif

#include "mktime-internal.h"

#if !defined _LIBC && (NEED_MKTIME_WORKING || NEED_MKTIME_WINDOWS)
static void
my_tzset (void)
{
# if NEED_MKTIME_WINDOWS
  /* Rectify the value of the environment variable TZ.
     There are four possible kinds of such values:
       - Traditional US time zone names, e.g. "PST8PDT".  Syntax: see
         <https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/tzset>
       - Time zone names based on geography, that contain one or more
         slashes, e.g. "Europe/Moscow".
       - Time zone names based on geography, without slashes, e.g.
         "Singapore".
       - Time zone names that contain explicit DST rules.  Syntax: see
         <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap08.html#tag_08_03>
     The Microsoft CRT understands only the first kind.  It produces incorrect
     results if the value of TZ is of the other kinds.
     But in a Cygwin environment, /etc/profile.d/tzset.sh sets TZ to a value
     of the second kind for most geographies, or of the first kind in a few
     other geographies.  If it is of the second kind, neutralize it.  For the
     Microsoft CRT, an absent or empty TZ means the time zone that the user
     has set in the Windows Control Panel.
     If the value of TZ is of the third or fourth kind -- Cygwin programs
     understand these syntaxes as well --, it does not matter whether we
     neutralize it or not, since these values occur only when a Cygwin user
     has set TZ explicitly; this case is 1. rare and 2. under the user's
     responsibility.  */
  const char *tz = getenv ("TZ");
  if (tz != NULL && strchr (tz, '/') != NULL)
    _putenv ("TZ=");
# elif HAVE_TZSET
  tzset ();
# endif
}
# undef __tzset
# define __tzset() my_tzset ()
#endif

#if defined _LIBC || NEED_MKTIME_WORKING || NEED_MKTIME_INTERNAL

/* A signed type that can represent an integer number of years
   multiplied by four times the number of seconds in a year.  It is
   needed when converting a tm_year value times the number of seconds
   in a year.  The factor of four comes because these products need
   to be subtracted from each other, and sometimes with an offset
   added to them, and then with another timestamp added, without
   worrying about overflow.

   Much of the code uses long_int to represent __time64_t values, to
   lessen the hassle of dealing with platforms where __time64_t is
   unsigned, and because long_int should suffice to represent all
   __time64_t values that mktime can generate even on platforms where
   __time64_t is wider than the int components of struct tm.  */

#if INT_MAX <= LONG_MAX / 4 / 366 / 24 / 60 / 60
typedef long int long_int;
#else
typedef long long int long_int;
#endif
verify (INT_MAX <= TYPE_MAXIMUM (long_int) / 4 / 366 / 24 / 60 / 60);

/* Shift A right by B bits portably, by dividing A by 2**B and
   truncating towards minus infinity.  B should be in the range 0 <= B
   <= LONG_INT_BITS - 2, where LONG_INT_BITS is the number of useful
   bits in a long_int.  LONG_INT_BITS is at least 32.

   ISO C99 says that A >> B is implementation-defined if A < 0.  Some
   implementations (e.g., UNICOS 9.0 on a Cray Y-MP EL) don't shift
   right in the usual way when A < 0, so SHR falls back on division if
   ordinary A >> B doesn't seem to be the usual signed shift.  */

static long_int
shr (long_int a, int b)
{
  long_int one = 1;
  return (-one >> 1 == -1
	  ? a >> b
	  : (a + (a < 0)) / (one << b) - (a < 0));
}

/* Bounds for the intersection of __time64_t and long_int.  */

static long_int const mktime_min
  = ((TYPE_SIGNED (__time64_t)
      && TYPE_MINIMUM (__time64_t) < TYPE_MINIMUM (long_int))
     ? TYPE_MINIMUM (long_int) : TYPE_MINIMUM (__time64_t));
static long_int const mktime_max
  = (TYPE_MAXIMUM (long_int) < TYPE_MAXIMUM (__time64_t)
     ? TYPE_MAXIMUM (long_int) : TYPE_MAXIMUM (__time64_t));

#define EPOCH_YEAR 1970
#define TM_YEAR_BASE 1900
verify (TM_YEAR_BASE % 100 == 0);

/* Is YEAR + TM_YEAR_BASE a leap year?  */
static bool
leapyear (long_int year)
{
  /* Don't add YEAR to TM_YEAR_BASE, as that might overflow.
     Also, work even if YEAR is negative.  */
  return
    ((year & 3) == 0
     && (year % 100 != 0
	 || ((year / 100) & 3) == (- (TM_YEAR_BASE / 100) & 3)));
}

/* How many days come before each month (0-12).  */
#ifndef _LIBC
static
#endif
const unsigned short int __mon_yday[2][13] =
  {
    /* Normal years.  */
    { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 },
    /* Leap years.  */
    { 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 }
  };


/* Do the values A and B differ according to the rules for tm_isdst?
   A and B differ if one is zero and the other positive.  */
static bool
isdst_differ (int a, int b)
{
  return (!a != !b) && (0 <= a) && (0 <= b);
}

/* Return an integer value measuring (YEAR1-YDAY1 HOUR1:MIN1:SEC1) -
   (YEAR0-YDAY0 HOUR0:MIN0:SEC0) in seconds, assuming that the clocks
   were not adjusted between the timestamps.

   The YEAR values uses the same numbering as TP->tm_year.  Values
   need not be in the usual range.  However, YEAR1 - YEAR0 must not
   overflow even when multiplied by three times the number of seconds
   in a year, and likewise for YDAY1 - YDAY0 and three times the
   number of seconds in a day.  */

static long_int
ydhms_diff (long_int year1, long_int yday1, int hour1, int min1, int sec1,
	    int year0, int yday0, int hour0, int min0, int sec0)
{
  verify (-1 / 2 == 0);

  /* Compute intervening leap days correctly even if year is negative.
     Take care to avoid integer overflow here.  */
  int a4 = shr (year1, 2) + shr (TM_YEAR_BASE, 2) - ! (year1 & 3);
  int b4 = shr (year0, 2) + shr (TM_YEAR_BASE, 2) - ! (year0 & 3);
  int a100 = (a4 + (a4 < 0)) / 25 - (a4 < 0);
  int b100 = (b4 + (b4 < 0)) / 25 - (b4 < 0);
  int a400 = shr (a100, 2);
  int b400 = shr (b100, 2);
  int intervening_leap_days = (a4 - b4) - (a100 - b100) + (a400 - b400);

  /* Compute the desired time without overflowing.  */
  long_int years = year1 - year0;
  long_int days = 365 * years + yday1 - yday0 + intervening_leap_days;
  long_int hours = 24 * days + hour1 - hour0;
  long_int minutes = 60 * hours + min1 - min0;
  long_int seconds = 60 * minutes + sec1 - sec0;
  return seconds;
}

/* Return the average of A and B, even if A + B would overflow.
   Round toward positive infinity.  */
static long_int
long_int_avg (long_int a, long_int b)
{
  return shr (a, 1) + shr (b, 1) + ((a | b) & 1);
}

/* Return a long_int value corresponding to (YEAR-YDAY HOUR:MIN:SEC)
   minus *TP seconds, assuming no clock adjustments occurred between
   the two timestamps.

   YEAR and YDAY must not be so large that multiplying them by three times the
   number of seconds in a year (or day, respectively) would overflow long_int.
   *TP should be in the usual range.  */
static long_int
tm_diff (long_int year, long_int yday, int hour, int min, int sec,
	 struct tm const *tp)
{
  return ydhms_diff (year, yday, hour, min, sec,
		     tp->tm_year, tp->tm_yday,
		     tp->tm_hour, tp->tm_min, tp->tm_sec);
}

/* Use CONVERT to convert T to a struct tm value in *TM.  T must be in
   range for __time64_t.  Return TM if successful, NULL (setting errno) on
   failure.  */
static struct tm *
convert_time (struct tm *(*convert) (const __time64_t *, struct tm *),
	      long_int t, struct tm *tm)
{
  __time64_t x = t;
  return convert (&x, tm);
}

/* Use CONVERT to convert *T to a broken down time in *TP.
   If *T is out of range for conversion, adjust it so that
   it is the nearest in-range value and then convert that.
   A value is in range if it fits in both __time64_t and long_int.
   Return TP on success, NULL (setting errno) on failure.  */
static struct tm *
ranged_convert (struct tm *(*convert) (const __time64_t *, struct tm *),
		long_int *t, struct tm *tp)
{
  long_int t1 = (*t < mktime_min ? mktime_min
		 : *t <= mktime_max ? *t : mktime_max);
  struct tm *r = convert_time (convert, t1, tp);
  if (r)
    {
      *t = t1;
      return r;
    }
  if (errno != EOVERFLOW)
    return NULL;

  long_int bad = t1;
  long_int ok = 0;
  struct tm oktm; oktm.tm_sec = -1;

  /* BAD is a known out-of-range value, and OK is a known in-range one.
     Use binary search to narrow the range between BAD and OK until
     they differ by 1.  */
  while (true)
    {
      long_int mid = long_int_avg (ok, bad);
      if (mid == ok || mid == bad)
	break;
      if (convert_time (convert, mid, tp))
	ok = mid, oktm = *tp;
      else if (errno != EOVERFLOW)
	return NULL;
      else
	bad = mid;
    }

  if (oktm.tm_sec < 0)
    return NULL;
  *t = ok;
  *tp = oktm;
  return tp;
}


/* Convert *TP to a __time64_t value, inverting
   the monotonic and mostly-unit-linear conversion function CONVERT.
   Use *OFFSET to keep track of a guess at the offset of the result,
   compared to what the result would be for UTC without leap seconds.
   If *OFFSET's guess is correct, only one CONVERT call is needed.
   If successful, set *TP to the canonicalized struct tm;
   otherwise leave *TP alone, return ((time_t) -1) and set errno.
   This function is external because it is used also by timegm.c.  */
__time64_t
__mktime_internal (struct tm *tp,
		   struct tm *(*convert) (const __time64_t *, struct tm *),
		   mktime_offset_t *offset)
{
  struct tm tm;

  /* The maximum number of probes (calls to CONVERT) should be enough
     to handle any combinations of time zone rule changes, solar time,
     leap seconds, and oscillations around a spring-forward gap.
     POSIX.1 prohibits leap seconds, but some hosts have them anyway.  */
  int remaining_probes = 6;

  /* Time requested.  Copy it in case CONVERT modifies *TP; this can
     occur if TP is localtime's returned value and CONVERT is localtime.  */
  int sec = tp->tm_sec;
  int min = tp->tm_min;
  int hour = tp->tm_hour;
  int mday = tp->tm_mday;
  int mon = tp->tm_mon;
  int year_requested = tp->tm_year;
  int isdst = tp->tm_isdst;

  /* 1 if the previous probe was DST.  */
  int dst2 = 0;

  /* Ensure that mon is in range, and set year accordingly.  */
  int mon_remainder = mon % 12;
  int negative_mon_remainder = mon_remainder < 0;
  int mon_years = mon / 12 - negative_mon_remainder;
  long_int lyear_requested = year_requested;
  long_int year = lyear_requested + mon_years;

  /* The other values need not be in range:
     the remaining code handles overflows correctly.  */

  /* Calculate day of year from year, month, and day of month.
     The result need not be in range.  */
  int mon_yday = ((__mon_yday[leapyear (year)]
		   [mon_remainder + 12 * negative_mon_remainder])
		  - 1);
  long_int lmday = mday;
  long_int yday = mon_yday + lmday;

  mktime_offset_t off = *offset;
  int negative_offset_guess;

  int sec_requested = sec;

  if (LEAP_SECONDS_POSSIBLE)
    {
      /* Handle out-of-range seconds specially,
	 since ydhms_diff assumes every minute has 60 seconds.  */
      if (sec < 0)
	sec = 0;
      if (59 < sec)
	sec = 59;
    }

  /* Invert CONVERT by probing.  First assume the same offset as last
     time.  */

  INT_SUBTRACT_WRAPV (0, off, &negative_offset_guess);
  long_int t0 = ydhms_diff (year, yday, hour, min, sec,
			    EPOCH_YEAR - TM_YEAR_BASE, 0, 0, 0,
			    negative_offset_guess);
  long_int t = t0, t1 = t0, t2 = t0;

  /* Repeatedly use the error to improve the guess.  */

  while (true)
    {
      if (! ranged_convert (convert, &t, &tm))
	return -1;
      long_int dt = tm_diff (year, yday, hour, min, sec, &tm);
      if (dt == 0)
	break;

      if (t == t1 && t != t2
	  && (tm.tm_isdst < 0
	      || (isdst < 0
		  ? dst2 <= (tm.tm_isdst != 0)
		  : (isdst != 0) != (tm.tm_isdst != 0))))
	/* We can't possibly find a match, as we are oscillating
	   between two values.  The requested time probably falls
	   within a spring-forward gap of size DT.  Follow the common
	   practice in this case, which is to return a time that is DT
	   away from the requested time, preferring a time whose
	   tm_isdst differs from the requested value.  (If no tm_isdst
	   was requested and only one of the two values has a nonzero
	   tm_isdst, prefer that value.)  In practice, this is more
	   useful than returning -1.  */
	goto offset_found;

      remaining_probes--;
      if (remaining_probes == 0)
	{
	  __set_errno (EOVERFLOW);
	  return -1;
	}

      t1 = t2, t2 = t, t += dt, dst2 = tm.tm_isdst != 0;
    }

  /* We have a match.  Check whether tm.tm_isdst has the requested
     value, if any.  */
  if (isdst_differ (isdst, tm.tm_isdst))
    {
      /* tm.tm_isdst has the wrong value.  Look for a neighboring
	 time with the right value, and use its UTC offset.

	 Heuristic: probe the adjacent timestamps in both directions,
	 looking for the desired isdst.  This should work for all real
	 time zone histories in the tz database.  */

      /* Distance between probes when looking for a DST boundary.  In
	 tzdata2003a, the shortest period of DST is 601200 seconds
	 (e.g., America/Recife starting 2000-10-08 01:00), and the
	 shortest period of non-DST surrounded by DST is 694800
	 seconds (Africa/Tunis starting 1943-04-17 01:00).  Use the
	 minimum of these two values, so we don't miss these short
	 periods when probing.  */
      int stride = 601200;

      /* The longest period of DST in tzdata2003a is 536454000 seconds
	 (e.g., America/Jujuy starting 1946-10-01 01:00).  The longest
	 period of non-DST is much longer, but it makes no real sense
	 to search for more than a year of non-DST, so use the DST
	 max.  */
      int duration_max = 536454000;

      /* Search in both directions, so the maximum distance is half
	 the duration; add the stride to avoid off-by-1 problems.  */
      int delta_bound = duration_max / 2 + stride;

      int delta, direction;

      for (delta = stride; delta < delta_bound; delta += stride)
	for (direction = -1; direction <= 1; direction += 2)
	  {
	    long_int ot;
	    if (! INT_ADD_WRAPV (t, delta * direction, &ot))
	      {
		struct tm otm;
		if (! ranged_convert (convert, &ot, &otm))
		  return -1;
		if (! isdst_differ (isdst, otm.tm_isdst))
		  {
		    /* We found the desired tm_isdst.
		       Extrapolate back to the desired time.  */
		    long_int gt = ot + tm_diff (year, yday, hour, min, sec,
						&otm);
		    if (mktime_min <= gt && gt <= mktime_max)
		      {
			if (convert_time (convert, gt, &tm))
			  {
			    t = gt;
			    goto offset_found;
			  }
			if (errno != EOVERFLOW)
			  return -1;
		      }
		  }
	      }
	  }

      __set_errno (EOVERFLOW);
      return -1;
    }

 offset_found:
  /* Set *OFFSET to the low-order bits of T - T0 - NEGATIVE_OFFSET_GUESS.
     This is just a heuristic to speed up the next mktime call, and
     correctness is unaffected if integer overflow occurs here.  */
  INT_SUBTRACT_WRAPV (t, t0, offset);
  INT_SUBTRACT_WRAPV (*offset, negative_offset_guess, offset);

  if (LEAP_SECONDS_POSSIBLE && sec_requested != tm.tm_sec)
    {
      /* Adjust time to reflect the tm_sec requested, not the normalized value.
	 Also, repair any damage from a false match due to a leap second.  */
      long_int sec_adjustment = sec == 0 && tm.tm_sec == 60;
      sec_adjustment -= sec;
      sec_adjustment += sec_requested;
      if (INT_ADD_WRAPV (t, sec_adjustment, &t)
	  || ! (mktime_min <= t && t <= mktime_max))
	{
	  __set_errno (EOVERFLOW);
	  return -1;
	}
      if (! convert_time (convert, t, &tm))
	return -1;
    }

  *tp = tm;
  return t;
}

#endif /* _LIBC || NEED_MKTIME_WORKING || NEED_MKTIME_INTERNAL */

#if defined _LIBC || NEED_MKTIME_WORKING || NEED_MKTIME_WINDOWS

/* Convert *TP to a __time64_t value.  */
__time64_t
__mktime64 (struct tm *tp)
{
  /* POSIX.1 8.1.1 requires that whenever mktime() is called, the
     time zone names contained in the external variable 'tzname' shall
     be set as if the tzset() function had been called.  */
  __tzset ();

# if defined _LIBC || NEED_MKTIME_WORKING
  static mktime_offset_t localtime_offset;
  return __mktime_internal (tp, __localtime64_r, &localtime_offset);
# else
#  undef mktime
  return mktime (tp);
# endif
}
#endif /* _LIBC || NEED_MKTIME_WORKING || NEED_MKTIME_WINDOWS */

#if defined _LIBC && __TIMESIZE != 64

libc_hidden_def (__mktime64)

time_t
mktime (struct tm *tp)
{
  struct tm tm = *tp;
  __time64_t t = __mktime64 (&tm);
  if (in_time_t_range (t))
    {
      *tp = tm;
      return t;
    }
  else
    {
      __set_errno (EOVERFLOW);
      return -1;
    }
}

#endif

weak_alias (mktime, timelocal)
libc_hidden_def (mktime)
libc_hidden_weak (timelocal)
