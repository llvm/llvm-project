/* Timer test using the monotonic clock.  */

#include <time.h>
#include <unistd.h>

#if defined CLOCK_MONOTONIC && defined _POSIX_MONOTONIC_CLOCK

# define TEST_CLOCK	CLOCK_MONOTONIC
# define TEST_CLOCK_MISSING(clock) \
  (setup_test () ? "CLOCK_MONOTONIC" : NULL)

# include <stdio.h>

static int
setup_test (void)
{
  if (sysconf (_SC_MONOTONIC_CLOCK) <= 0)
    return 1;

  /* The user-level timers implementation doesn't support CLOCK_MONOTONIC,
     even though sysconf claims it will.  */
  timer_t t;
  if (timer_create (TEST_CLOCK, NULL, &t) != 0)
    {
      printf ("timer_create: %m\n");
      return 1;
    }
  timer_delete (t);

  return 0;
}

# include "tst-timer4.c"

#else
# define TEST_FUNCTION	0
# include "../test-skeleton.c"
#endif
