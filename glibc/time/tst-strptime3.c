#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


static int
do_test (void)
{
  int result = 0;
  struct tm tm;

  memset (&tm, 0xaa, sizeof (tm));

  /* Test we don't crash on uninitialized struct tm.
     Some fields might contain bogus values until everything
     needed is initialized, but we shouldn't crash.  */
  if (strptime ("2007", "%Y", &tm) == NULL
      || strptime ("12", "%d", &tm) == NULL
      || strptime ("Feb", "%b", &tm) == NULL
      || strptime ("13", "%M", &tm) == NULL
      || strptime ("21", "%S", &tm) == NULL
      || strptime ("16", "%H", &tm) == NULL)
    {
      puts ("strptimes failed");
      result = 1;
    }

  if (tm.tm_sec != 21 || tm.tm_min != 13 || tm.tm_hour != 16
      || tm.tm_mday != 12 || tm.tm_mon != 1 || tm.tm_year != 107
      || tm.tm_wday != 1 || tm.tm_yday != 42)
    {
      puts ("unexpected tm content");
      result = 1;
    }

  if (strptime ("8", "%d", &tm) == NULL)
    {
      puts ("strptime failed");
      result = 1;
    }

  if (tm.tm_sec != 21 || tm.tm_min != 13 || tm.tm_hour != 16
      || tm.tm_mday != 8 || tm.tm_mon != 1 || tm.tm_year != 107
      || tm.tm_wday != 4 || tm.tm_yday != 38)
    {
      puts ("unexpected tm content");
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
