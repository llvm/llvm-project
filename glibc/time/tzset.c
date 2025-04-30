/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <ctype.h>
#include <libc-lock.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <timezone/tzfile.h>

#define SECSPERDAY ((__time64_t) 86400)

char *__tzname[2] = { (char *) "GMT", (char *) "GMT" };
int __daylight = 0;
long int __timezone = 0L;

weak_alias (__tzname, tzname)
weak_alias (__daylight, daylight)
weak_alias (__timezone, timezone)

/* This locks all the state variables in tzfile.c and this file.  */
__libc_lock_define_initialized (static, tzset_lock)

/* This structure contains all the information about a
   timezone given in the POSIX standard TZ envariable.  */
typedef struct
  {
    const char *name;

    /* When to change.  */
    enum { J0, J1, M } type;	/* Interpretation of:  */
    unsigned short int m, n, d;	/* Month, week, day.  */
    int secs;			/* Time of day.  */

    int offset;			/* Seconds east of GMT (west if < 0).  */

    /* We cache the computed time of change for a
       given year so we don't have to recompute it.  */
    __time64_t change;	/* When to change to this zone.  */
    int computed_for;	/* Year above is computed for.  */
  } tz_rule;

/* tz_rules[0] is standard, tz_rules[1] is daylight.  */
static tz_rule tz_rules[2];


static void compute_change (tz_rule *rule, int year) __THROW;
static void tzset_internal (int always);

/* List of buffers containing time zone strings. */
struct tzstring_l
{
  struct tzstring_l *next;
  size_t len;  /* strlen(data) - doesn't count terminating NUL! */
  char data[0];
};

static struct tzstring_l *tzstring_list;

/* Allocate a permanent home for the first LEN characters of S.  It
   will never be moved or deallocated, but may share space with other
   strings.  Don't modify the returned string. */
static char *
__tzstring_len (const char *s, size_t len)
{
  char *p;
  struct tzstring_l *t, *u, *new;

  /* Walk the list and look for a match.  If this string is the same
     as the end of an already-allocated string, it can share space. */
  for (u = t = tzstring_list; t; u = t, t = t->next)
    if (len <= t->len)
      {
	p = &t->data[t->len - len];
	if (memcmp (s, p, len) == 0)
	  return p;
      }

  /* Not found; allocate a new buffer. */
  new = malloc (sizeof (struct tzstring_l) + len + 1);
  if (!new)
    return NULL;

  new->next = NULL;
  new->len = len;
  memcpy (new->data, s, len);
  new->data[len] = '\0';

  if (u)
    u->next = new;
  else
    tzstring_list = new;

  return new->data;
}

/* Allocate a permanent home for S.  It will never be moved or
   deallocated, but may share space with other strings.  Don't modify
   the returned string. */
char *
__tzstring (const char *s)
{
  return __tzstring_len (s, strlen (s));
}

static char *old_tz;

static void
update_vars (void)
{
  __daylight = tz_rules[0].offset != tz_rules[1].offset;
  __timezone = -tz_rules[0].offset;
  __tzname[0] = (char *) tz_rules[0].name;
  __tzname[1] = (char *) tz_rules[1].name;
}


static unsigned int
compute_offset (unsigned int ss, unsigned int mm, unsigned int hh)
{
  if (ss > 59)
    ss = 59;
  if (mm > 59)
    mm = 59;
  if (hh > 24)
    hh = 24;
  return ss + mm * 60 + hh * 60 * 60;
}

/* Parses the time zone name at *TZP, and writes a pointer to an
   interned string to tz_rules[WHICHRULE].name.  On success, advances
   *TZP, and returns true.  Returns false otherwise.  */
static bool
parse_tzname (const char **tzp, int whichrule)
{
  const char *start = *tzp;
  const char *p = start;
  while (('a' <= *p && *p <= 'z')
	 || ('A' <= *p && *p <= 'Z'))
      ++p;
  size_t len = p - start;
  if (len < 3)
    {
      p = *tzp;
      if (__glibc_unlikely (*p++ != '<'))
	return false;
      start = p;
      while (('a' <= *p && *p <= 'z')
	     || ('A' <= *p && *p <= 'Z')
	     || ('0' <= *p && *p <= '9')
	     || *p == '+' || *p == '-')
	++p;
      len = p - start;
      if (*p++ != '>' || len < 3)
	return false;
    }

  const char *name = __tzstring_len (start, len);
  if (name == NULL)
    return false;
  tz_rules[whichrule].name = name;

  *tzp = p;
  return true;
}

/* Parses the time zone offset at *TZP, and writes it to
   tz_rules[WHICHRULE].offset.  Returns true if the parse was
   successful.  */
static bool
parse_offset (const char **tzp, int whichrule)
{
  const char *tz = *tzp;
  if (whichrule == 0
      && (*tz == '\0' || (*tz != '+' && *tz != '-' && !isdigit (*tz))))
    return false;

  int sign;
  if (*tz == '-' || *tz == '+')
    sign = *tz++ == '-' ? 1 : -1;
  else
    sign = -1;
  *tzp = tz;

  unsigned short int hh;
  unsigned short mm = 0;
  unsigned short ss = 0;
  int consumed = 0;
  if (sscanf (tz, "%hu%n:%hu%n:%hu%n",
	      &hh, &consumed, &mm, &consumed, &ss, &consumed) > 0)
    tz_rules[whichrule].offset = sign * compute_offset (ss, mm, hh);
  else
    /* Nothing could be parsed. */
    if (whichrule == 0)
      {
	/* Standard time defaults to offset zero.  */
	tz_rules[0].offset = 0;
	return false;
      }
      else
	/* DST defaults to one hour later than standard time.  */
	tz_rules[1].offset = tz_rules[0].offset + (60 * 60);
  *tzp = tz + consumed;
  return true;
}

/* Parses the standard <-> DST rules at *TZP.  Updates
   tz_rule[WHICHRULE].  On success, advances *TZP and returns true.
   Otherwise, returns false.  */
static bool
parse_rule (const char **tzp, int whichrule)
{
  const char *tz = *tzp;
  tz_rule *tzr = &tz_rules[whichrule];

  /* Ignore comma to support string following the incorrect
     specification in early POSIX.1 printings.  */
  tz += *tz == ',';

  /* Get the date of the change.  */
  if (*tz == 'J' || isdigit (*tz))
    {
      char *end;
      tzr->type = *tz == 'J' ? J1 : J0;
      if (tzr->type == J1 && !isdigit (*++tz))
	return false;
      unsigned long int d = strtoul (tz, &end, 10);
      if (end == tz || d > 365)
	return false;
      if (tzr->type == J1 && d == 0)
	return false;
      tzr->d = d;
      tz = end;
    }
  else if (*tz == 'M')
    {
      tzr->type = M;
      int consumed;
      if (sscanf (tz, "M%hu.%hu.%hu%n",
		  &tzr->m, &tzr->n, &tzr->d, &consumed) != 3
	  || tzr->m < 1 || tzr->m > 12
	  || tzr->n < 1 || tzr->n > 5 || tzr->d > 6)
	return false;
      tz += consumed;
    }
  else if (*tz == '\0')
    {
      /* Daylight time rules in the U.S. are defined in the U.S. Code,
	 Title 15, Chapter 6, Subchapter IX - Standard Time.  These
	 dates were established by Congress in the Energy Policy Act
	 of 2005 [Pub. L. no. 109-58, 119 Stat 594 (2005)].
	 Below is the equivalent of "M3.2.0,M11.1.0" [/2 not needed
	 since 2:00AM is the default].  */
      tzr->type = M;
      if (tzr == &tz_rules[0])
	{
	  tzr->m = 3;
	  tzr->n = 2;
	  tzr->d = 0;
	}
      else
	{
	  tzr->m = 11;
	  tzr->n = 1;
	  tzr->d = 0;
	}
    }
  else
    return false;

  if (*tz != '\0' && *tz != '/' && *tz != ',')
    return false;
  else if (*tz == '/')
    {
      /* Get the time of day of the change.  */
      int negative;
      ++tz;
      if (*tz == '\0')
	return false;
      negative = *tz == '-';
      tz += negative;
      /* Default to 2:00 AM.  */
      unsigned short hh = 2;
      unsigned short mm = 0;
      unsigned short ss = 0;
      int consumed = 0;
      sscanf (tz, "%hu%n:%hu%n:%hu%n",
	      &hh, &consumed, &mm, &consumed, &ss, &consumed);;
      tz += consumed;
      tzr->secs = (negative ? -1 : 1) * ((hh * 60 * 60) + (mm * 60) + ss);
    }
  else
    /* Default to 2:00 AM.  */
    tzr->secs = 2 * 60 * 60;

  tzr->computed_for = -1;
  *tzp = tz;
  return true;
}

/* Parse the POSIX TZ-style string.  */
void
__tzset_parse_tz (const char *tz)
{
  /* Clear out old state and reset to unnamed UTC.  */
  memset (tz_rules, '\0', sizeof tz_rules);
  tz_rules[0].name = tz_rules[1].name = "";

  /* Get the standard timezone name.  */
  if (parse_tzname (&tz, 0) && parse_offset (&tz, 0))
    {
      /* Get the DST timezone name (if any).  */
      if (*tz != '\0')
	{
	  if (parse_tzname (&tz, 1))
	    {
	      parse_offset (&tz, 1);
	      if (*tz == '\0' || (tz[0] == ',' && tz[1] == '\0'))
		{
		  /* There is no rule.  See if there is a default rule
		     file.  */
		  __tzfile_default (tz_rules[0].name, tz_rules[1].name,
				    tz_rules[0].offset, tz_rules[1].offset);
		  if (__use_tzfile)
		    {
		      free (old_tz);
		      old_tz = NULL;
		      return;
		    }
		}
	    }
	  /* Figure out the standard <-> DST rules.  */
	  if (parse_rule (&tz, 0))
	    parse_rule (&tz, 1);
	}
      else
	{
	  /* There is no DST.  */
	  tz_rules[1].name = tz_rules[0].name;
	  tz_rules[1].offset = tz_rules[0].offset;
	}
    }

  update_vars ();
}

/* Interpret the TZ envariable.  */
static void
tzset_internal (int always)
{
  static int is_initialized;
  const char *tz;

  if (is_initialized && !always)
    return;
  is_initialized = 1;

  /* Examine the TZ environment variable.  */
  tz = getenv ("TZ");
  if (tz && *tz == '\0')
    /* User specified the empty string; use UTC explicitly.  */
    tz = "Universal";

  /* A leading colon means "implementation defined syntax".
     We ignore the colon and always use the same algorithm:
     try a data file, and if none exists parse the 1003.1 syntax.  */
  if (tz && *tz == ':')
    ++tz;

  /* Check whether the value changed since the last run.  */
  if (old_tz != NULL && tz != NULL && strcmp (tz, old_tz) == 0)
    /* No change, simply return.  */
    return;

  if (tz == NULL)
    /* No user specification; use the site-wide default.  */
    tz = TZDEFAULT;

  tz_rules[0].name = NULL;
  tz_rules[1].name = NULL;

  /* Save the value of `tz'.  */
  free (old_tz);
  old_tz = tz ? __strdup (tz) : NULL;

  /* Try to read a data file.  */
  __tzfile_read (tz, 0, NULL);
  if (__use_tzfile)
    return;

  /* No data file found.  Default to UTC if nothing specified.  */

  if (tz == NULL || *tz == '\0'
      || (TZDEFAULT != NULL && strcmp (tz, TZDEFAULT) == 0))
    {
      memset (tz_rules, '\0', sizeof tz_rules);
      tz_rules[0].name = tz_rules[1].name = "UTC";
      if (J0 != 0)
	tz_rules[0].type = tz_rules[1].type = J0;
      tz_rules[0].change = tz_rules[1].change = -1;
      update_vars ();
      return;
    }

  __tzset_parse_tz (tz);
}

/* Figure out the exact time (as a __time64_t) in YEAR
   when the change described by RULE will occur and
   put it in RULE->change, saving YEAR in RULE->computed_for.  */
static void
compute_change (tz_rule *rule, int year)
{
  __time64_t t;

  if (year != -1 && rule->computed_for == year)
    /* Operations on times in 2 BC will be slower.  Oh well.  */
    return;

  /* First set T to January 1st, 0:00:00 GMT in YEAR.  */
  if (year > 1970)
    t = ((year - 1970) * 365
	 + /* Compute the number of leapdays between 1970 and YEAR
	      (exclusive).  There is a leapday every 4th year ...  */
	 + ((year - 1) / 4 - 1970 / 4)
	 /* ... except every 100th year ... */
	 - ((year - 1) / 100 - 1970 / 100)
	 /* ... but still every 400th year.  */
	 + ((year - 1) / 400 - 1970 / 400)) * SECSPERDAY;
  else
    t = 0;

  switch (rule->type)
    {
    case J1:
      /* Jn - Julian day, 1 == January 1, 60 == March 1 even in leap years.
	 In non-leap years, or if the day number is 59 or less, just
	 add SECSPERDAY times the day number-1 to the time of
	 January 1, midnight, to get the day.  */
      t += (rule->d - 1) * SECSPERDAY;
      if (rule->d >= 60 && __isleap (year))
	t += SECSPERDAY;
      break;

    case J0:
      /* n - Day of year.
	 Just add SECSPERDAY times the day number to the time of Jan 1st.  */
      t += rule->d * SECSPERDAY;
      break;

    case M:
      /* Mm.n.d - Nth "Dth day" of month M.  */
      {
	unsigned int i;
	int d, m1, yy0, yy1, yy2, dow;
	const unsigned short int *myday =
	  &__mon_yday[__isleap (year)][rule->m];

	/* First add SECSPERDAY for each day in months before M.  */
	t += myday[-1] * SECSPERDAY;

	/* Use Zeller's Congruence to get day-of-week of first day of month. */
	m1 = (rule->m + 9) % 12 + 1;
	yy0 = (rule->m <= 2) ? (year - 1) : year;
	yy1 = yy0 / 100;
	yy2 = yy0 % 100;
	dow = ((26 * m1 - 2) / 10 + 1 + yy2 + yy2 / 4 + yy1 / 4 - 2 * yy1) % 7;
	if (dow < 0)
	  dow += 7;

	/* DOW is the day-of-week of the first day of the month.  Get the
	   day-of-month (zero-origin) of the first DOW day of the month.  */
	d = rule->d - dow;
	if (d < 0)
	  d += 7;
	for (i = 1; i < rule->n; ++i)
	  {
	    if (d + 7 >= (int) myday[0] - myday[-1])
	      break;
	    d += 7;
	  }

	/* D is the day-of-month (zero-origin) of the day we want.  */
	t += d * SECSPERDAY;
      }
      break;
    }

  /* T is now the Epoch-relative time of 0:00:00 GMT on the day we want.
     Just add the time of day and local offset from GMT, and we're done.  */

  rule->change = t - rule->offset + rule->secs;
  rule->computed_for = year;
}


/* Figure out the correct timezone for TM and set `__tzname',
   `__timezone', and `__daylight' accordingly.  */
void
__tz_compute (__time64_t timer, struct tm *tm, int use_localtime)
{
  compute_change (&tz_rules[0], 1900 + tm->tm_year);
  compute_change (&tz_rules[1], 1900 + tm->tm_year);

  if (use_localtime)
    {
      int isdst;

      /* We have to distinguish between northern and southern
	 hemisphere.  For the latter the daylight saving time
	 ends in the next year.  */
      if (__builtin_expect (tz_rules[0].change
			    > tz_rules[1].change, 0))
	isdst = (timer < tz_rules[1].change
		 || timer >= tz_rules[0].change);
      else
	isdst = (timer >= tz_rules[0].change
		 && timer < tz_rules[1].change);
      tm->tm_isdst = isdst;
      tm->tm_zone = __tzname[isdst];
      tm->tm_gmtoff = tz_rules[isdst].offset;
    }
}

/* Reinterpret the TZ environment variable and set `tzname'.  */
#undef tzset

void
__tzset (void)
{
  __libc_lock_lock (tzset_lock);

  tzset_internal (1);

  if (!__use_tzfile)
    {
      /* Set `tzname'.  */
      __tzname[0] = (char *) tz_rules[0].name;
      __tzname[1] = (char *) tz_rules[1].name;
    }

  __libc_lock_unlock (tzset_lock);
}
weak_alias (__tzset, tzset)

/* Return the `struct tm' representation of TIMER in the local timezone.
   Use local time if USE_LOCALTIME is nonzero, UTC otherwise.  */
struct tm *
__tz_convert (__time64_t timer, int use_localtime, struct tm *tp)
{
  long int leap_correction;
  int leap_extra_secs;

  __libc_lock_lock (tzset_lock);

  /* Update internal database according to current TZ setting.
     POSIX.1 8.3.7.2 says that localtime_r is not required to set tzname.
     This is a good idea since this allows at least a bit more parallelism.  */
  tzset_internal (tp == &_tmbuf && use_localtime);

  if (__use_tzfile)
    __tzfile_compute (timer, use_localtime, &leap_correction,
		      &leap_extra_secs, tp);
  else
    {
      if (! __offtime (timer, 0, tp))
	tp = NULL;
      else
	__tz_compute (timer, tp, use_localtime);
      leap_correction = 0L;
      leap_extra_secs = 0;
    }

  __libc_lock_unlock (tzset_lock);

  if (tp)
    {
      if (! use_localtime)
	{
	  tp->tm_isdst = 0;
	  tp->tm_zone = "GMT";
	  tp->tm_gmtoff = 0L;
	}

      if (__offtime (timer, tp->tm_gmtoff - leap_correction, tp))
        tp->tm_sec += leap_extra_secs;
      else
	tp = NULL;
    }

  return tp;
}


libc_freeres_fn (free_mem)
{
  while (tzstring_list != NULL)
    {
      struct tzstring_l *old = tzstring_list;

      tzstring_list = tzstring_list->next;
      free (old);
    }
  free (old_tz);
  old_tz = NULL;
}
