#include <time.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool
equal_tm (struct tm const *t, struct tm const *u)
{
  return (t->tm_sec == u->tm_sec && t->tm_min == u->tm_min
	  && t->tm_hour == u->tm_hour && t->tm_mday == u->tm_mday
	  && t->tm_mon == u->tm_mon && t->tm_year == u->tm_year
	  && t->tm_wday == u->tm_wday && t->tm_yday == u->tm_yday
	  && t->tm_isdst == u->tm_isdst && t->tm_gmtoff == u->tm_gmtoff
	  && t->tm_zone == u->tm_zone);
}

static int
do_test (void)
{
  /* Calculate minimum time_t value.  This would be simpler with C11,
     which has _Generic, but we cannot assume C11.  It would also
     be simpler with intprops.h, which has TYPE_MINIMUM, but it's
     better not to use glibc internals.  */
  time_t time_t_min = -1;
  time_t_min = (0 < time_t_min ? 0
		: sizeof time_t_min == sizeof (long int) ? LONG_MIN
		: sizeof time_t_min == sizeof (long long int) ? LLONG_MIN
		: 1);
  if (time_t_min == 1)
    {
      printf ("unknown time type\n");
      return 1;
    }
  time_t ymin = time_t_min / 60 / 60 / 24 / 366;
  bool mktime_should_fail = ymin == 0 || INT_MIN + 1900 < ymin + 1970;

  struct tm tm0 = { .tm_year = INT_MIN, .tm_mday = 1, .tm_wday = -1 };
  struct tm tm = tm0;
  errno = 0;
  time_t t = mktime (&tm);
  long long int llt = t;
  bool mktime_failed = tm.tm_wday == tm0.tm_wday;

  if (mktime_failed)
    {
      if (! mktime_should_fail)
	{
	  printf ("mktime failed but should have succeeded\n");
	  return 1;
	}
      if (errno == 0)
	{
	  printf ("mktime failed without setting errno");
	  return 1;
	}
      if (t != (time_t) -1)
	{
	  printf ("mktime returned %lld but did not set tm_wday\n", llt);
	  return 1;
	}
      if (! equal_tm (&tm, &tm0))
	{
	  printf ("mktime (P) failed but modified *P\n");
	  return 1;
	}
    }
  else
    {
      if (mktime_should_fail)
	{
	  printf ("mktime succeeded but should have failed\n");
	  return 1;
	}
      struct tm *lt = localtime (&t);
      if (lt == NULL)
	{
	  printf ("mktime returned a value rejected by localtime\n");
	  return 1;
	}
      if (! equal_tm (lt, &tm))
	{
	  printf ("mktime result does not match localtime result\n");
	  return 1;
	}
    }
  return 0;
}

#define TIMEOUT 1000
#include "support/test-driver.c"
