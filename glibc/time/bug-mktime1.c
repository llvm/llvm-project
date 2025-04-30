#include <stdio.h>
#include <time.h>


static int
do_test (void)
{
  struct tm t2 = { 0, 0, 0, 1, 1, 2050 - 1900, 1, 1, 1 };
  time_t tt2 = mktime (&t2);
  printf ("%ld\n", (long int) tt2);
  if (sizeof (time_t) == 4 && tt2 != -1)
    return 1;
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
