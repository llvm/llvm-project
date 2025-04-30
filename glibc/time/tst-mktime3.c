/* Test program for mktime bugs with out-of-range tm_sec values.  */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

struct tm tests[] =
{
  { .tm_sec = -1, .tm_mday = 1, .tm_year = 104 },
  { .tm_sec = 65, .tm_min = 59, .tm_hour = 23, .tm_mday = 31,
    .tm_mon = 11, .tm_year = 101 }
};
struct tm expected[] =
{
  { .tm_sec = 59, .tm_min = 59, .tm_hour = 23, .tm_mday = 31,
    .tm_mon = 11, .tm_year = 103, .tm_wday = 3, .tm_yday = 364 },
  { .tm_sec = 5, .tm_mday = 1, .tm_year = 102, .tm_wday = 2 }
};

static int
do_test (void)
{
  setenv ("TZ", "UTC", 1);
  int i;
  for (i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      if (mktime (&tests[i]) < 0)
	{
	  printf ("mktime %d failed\n", i);
	  return 1;
	}
#define CHECK(name) \
      if (tests[i].name != expected[i].name)			\
	{							\
	  printf ("test %d " #name " got %d expected %d\n",	\
		  i, tests[i].name, expected[i].name);		\
	  return 1;						\
	}
      CHECK (tm_sec)
      CHECK (tm_min)
      CHECK (tm_hour)
      CHECK (tm_mday)
      CHECK (tm_mon)
      CHECK (tm_year)
      CHECK (tm_wday)
      CHECK (tm_yday)
      CHECK (tm_isdst)
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
