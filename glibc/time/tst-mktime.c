#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static int
do_test (void)
{
  struct tm time_str, *tm;
  time_t t;
  char daybuf[20];
  int result;

  time_str.tm_year = 2001 - 1900;
  time_str.tm_mon = 7 - 1;
  time_str.tm_mday = 4;
  time_str.tm_hour = 0;
  time_str.tm_min = 0;
  time_str.tm_sec = 1;
  time_str.tm_isdst = -1;

  if (mktime (&time_str) == -1)
    {
      (void) puts ("-unknown-");
      result = 1;
    }
  else
    {
      (void) strftime (daybuf, sizeof (daybuf), "%A", &time_str);
      (void) puts (daybuf);
      result = strcmp (daybuf, "Wednesday") != 0;
    }

  setenv ("TZ", "EST+5", 1);
#define EVENING69 1 * 60 * 60 + 2 * 60 + 29
  t = EVENING69;
  tm = localtime (&t);
  if (tm == NULL)
    {
      (void) puts ("localtime returned NULL");
      result = 1;
    }
  else
    {
      time_str = *tm;
      t = mktime (&time_str);
      if (t != EVENING69)
        {
          printf ("mktime returned %ld, expected %d\n",
		  (long) t, EVENING69);
	  result = 1;
        }
      else
        (void) puts ("Dec 31 1969 EST test passed");

      setenv ("TZ", "CET-1", 1);
      t = mktime (&time_str);
#define EVENING69_CET (EVENING69 - (5 - -1) * 60 * 60)
      if (t != EVENING69_CET)
        {
	  printf ("mktime returned %ld, expected %ld\n",
		  (long) t, (long) EVENING69_CET);
	  result = 1;
        }
      else
        (void) puts ("Dec 31 1969 CET test passed");
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
