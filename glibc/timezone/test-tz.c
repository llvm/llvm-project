#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

struct {
  const char *	env;
  time_t	expected;
} tests[] = {
  {"MST",	832935315},
  {"",		832910115},
  {":UTC",	832910115},
  {"UTC",	832910115},
  {"UTC0",	832910115}
};


int
main (int argc, char ** argv)
{
  int errors = 0;
  struct tm tm;
  time_t t;
  unsigned int i;

  memset (&tm, 0, sizeof (tm));
  tm.tm_isdst = 0;
  tm.tm_year  = 96;	/* years since 1900 */
  tm.tm_mon   = 4;
  tm.tm_mday  = 24;
  tm.tm_hour  =  3;
  tm.tm_min   = 55;
  tm.tm_sec   = 15;

  for (i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      setenv ("TZ", tests[i].env, 1);
      t = mktime (&tm);
      if (t != tests[i].expected)
	{
	  printf ("%s: flunked test %u (expected %lu, got %lu)\n",
		  argv[0], i, (long) tests[i].expected, (long) t);
	  ++errors;
	}
    }
  if (errors == 0)
    {
      puts ("No errors.");
      return EXIT_SUCCESS;
    }
  else
    {
      printf ("%d errors.\n", errors);
      return EXIT_FAILURE;
    }
}
