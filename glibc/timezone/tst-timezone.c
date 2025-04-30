/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 1998.

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

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

int failed = 0;

struct test_times
{
  const char *name;
  int daylight;
  int timezone;
  const char *tzname[2];
};

static const struct test_times tests[] =
{
  { "Europe/Amsterdam", 1, -3600, { "CET", "CEST" }},
  { "Europe/Berlin", 1, -3600, { "CET", "CEST" }},
  { "Europe/London", 1, 0, { "GMT", "BST" }},
  { "Universal", 0, 0, {"UTC", "UTC" }},
  { "Australia/Melbourne", 1, -36000, { "EST", "EST" }},
  { "America/Sao_Paulo", 1, 10800, {"BRT", "BRST" }},
  { "America/Chicago", 1, 21600, {"CST", "CDT" }},
  { "America/Indiana/Indianapolis", 1, 18000, {"EST", "EDT" }},
  { "America/Los_Angeles", 1, 28800, {"PST", "PDT" }},
  { "Pacific/Auckland", 1, -43200, { "NZST", "NZDT" }},
  { NULL, 0, 0 }
};

/* This string will be used for `putenv' calls.  */
char envstring[100];

static void
print_tzvars (void)
{
  printf ("tzname[0]: %s\n", tzname[0]);
  printf ("tzname[1]: %s\n", tzname[1]);
  printf ("daylight: %d\n", daylight);
  printf ("timezone: %ld\n", timezone);
}


static void
check_tzvars (const char *name, int dayl, int timez, const char *const tznam[])
{
  int i;

  if (daylight != dayl)
    {
      printf ("*** Timezone: %s, daylight is: %d but should be: %d\n",
	      name, daylight, dayl);
      ++failed;
    }
  if (timezone != timez)
    {
      printf ("*** Timezone: %s, timezone is: %ld but should be: %d\n",
	      name, timezone, timez);
      ++failed;
    }
  for (i = 0; i <= 1; ++i)
    if (strcmp (tzname[i], tznam[i]) != 0)
      {
	printf ("*** Timezone: %s, tzname[%d] is: %s but should be: %s\n",
		name, i, tzname[i], tznam[i]);
	++failed;
      }
}


static int
do_test (void)
{
  time_t t;
  const struct test_times *pt;
  char buf[BUFSIZ];

  /* This should be: Fri May 15 01:02:16 1998 (UTC).  */
  t = 895194136;
  printf ("We use this date: %s\n", asctime (gmtime (&t)));

  for (pt = tests; pt->name != NULL; ++pt)
    {
      /* Start with a known state */
      printf ("Checking timezone %s\n", pt->name);
      sprintf (buf, "TZ=%s", pt->name);
      if (putenv (buf))
	{
	  puts ("putenv failed.");
	  failed = 1;
	}
      tzset ();
      print_tzvars ();
      check_tzvars (pt->name, pt->daylight, pt->timezone, pt->tzname);

      /* calling localtime shouldn't make a difference */
      localtime (&t);
      print_tzvars ();
      check_tzvars (pt->name, pt->daylight, pt->timezone, pt->tzname);
    }

  /* From a post of Scott Harrington <seh4@ix.netcom.com> to the timezone
     mailing list.  */
  {
    struct tm tmBuf = {0, 0, 0, 10, 3, 98, 0, 0, -1};
    char buf[200];
    strcpy (envstring, "TZ=Europe/London");
    putenv (envstring);
    t = mktime (&tmBuf);
    snprintf (buf, sizeof (buf), "TZ=%s %jd %d %d %d %d %d %d %d %d %d",
	      getenv ("TZ"), (intmax_t) t,
	      tmBuf.tm_sec, tmBuf.tm_min, tmBuf.tm_hour,
	      tmBuf.tm_mday, tmBuf.tm_mon, tmBuf.tm_year,
	      tmBuf.tm_wday, tmBuf.tm_yday, tmBuf.tm_isdst);
    fputs (buf, stdout);
    puts (" should be");
    puts ("TZ=Europe/London 892162800 0 0 0 10 3 98 5 99 1");
    if (strcmp (buf, "TZ=Europe/London 892162800 0 0 0 10 3 98 5 99 1") != 0)
      {
	failed = 1;
	fputs (" FAILED ***", stdout);
      }
  }

  printf("\n");

  {
    struct tm tmBuf = {0, 0, 0, 10, 3, 98, 0, 0, -1};
    char buf[200];
    strcpy (envstring, "TZ=GMT");
    /* No putenv call needed!  */
    t = mktime (&tmBuf);
    snprintf (buf, sizeof (buf), "TZ=%s %jd %d %d %d %d %d %d %d %d %d",
	      getenv ("TZ"), (intmax_t) t,
	      tmBuf.tm_sec, tmBuf.tm_min, tmBuf.tm_hour,
	      tmBuf.tm_mday, tmBuf.tm_mon, tmBuf.tm_year,
	      tmBuf.tm_wday, tmBuf.tm_yday, tmBuf.tm_isdst);
    fputs (buf, stdout);
    puts (" should be");
    puts ("TZ=GMT 892166400 0 0 0 10 3 98 5 99 0");
    if (strcmp (buf, "TZ=GMT 892166400 0 0 0 10 3 98 5 99 0") != 0)
      {
	failed = 1;
	fputs (" FAILED ***", stdout);
      }
  }

  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
