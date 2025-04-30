#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <time.h>


static int
do_test (void)
{
  int result = 0;
  time_t t = time (NULL);
  struct tm *tp = localtime (&t);
  tp->tm_year = 10000 - 1900;
  char buf[1000];
  errno = 0;
  buf[26] = '\xff';
  char *s = asctime_r (tp, buf);
  if (s != NULL || errno != EOVERFLOW)
    {
      puts ("asctime_r did not fail correctly");
      result = 1;
    }
  if (buf[26] != '\xff')
    {
      puts ("asctime_r overwrote 27th byte in buffer");
      result = 1;
    }
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
