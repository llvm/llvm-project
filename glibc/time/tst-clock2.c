/* Test setting the monotonic clock.  */

#include <time.h>
#include <unistd.h>

#if defined CLOCK_MONOTONIC && defined _POSIX_MONOTONIC_CLOCK

# include <errno.h>
# include <stdio.h>

static int
do_test (void)
{
  if (sysconf (_SC_MONOTONIC_CLOCK) <= 0)
    return 0;

  struct timespec ts;
  if (clock_gettime (CLOCK_MONOTONIC, &ts) != 0)
    {
      puts ("clock_gettime(CLOCK_MONOTONIC) failed");
      return 1;
    }

  /* Setting the monotonic clock must fail.  */
  if (clock_settime (CLOCK_MONOTONIC, &ts) != -1)
    {
      puts ("clock_settime(CLOCK_MONOTONIC) did not fail");
      return 1;
    }
  if (errno != EINVAL)
    {
      printf ("clock_settime(CLOCK_MONOTONIC) set errno to %d, expected %d\n",
	      errno, EINVAL);
      return 1;
    }
  return 0;
}
# define TEST_FUNCTION do_test ()

#else
# define TEST_FUNCTION	0
#endif
#include "../test-skeleton.c"
