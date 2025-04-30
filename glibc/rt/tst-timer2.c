/* Test for crashing bugs when trying to create too many timers.  */

#include <stdio.h>
#include <time.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#if _POSIX_THREADS
# include <pthread.h>

void
thread (union sigval arg)
{
  puts ("Timeout");
}

int
do_test (void)
{
  int i, res;
  timer_t timerId;
  struct itimerspec itval;
  struct sigevent sigev;

  itval.it_interval.tv_sec = 2;
  itval.it_interval.tv_nsec = 0;
  itval.it_value.tv_sec = 2;
  itval.it_value.tv_nsec = 0;

  sigev.sigev_notify = SIGEV_THREAD;
  sigev.sigev_notify_function = thread;
  sigev.sigev_notify_attributes = NULL;
  sigev.sigev_value.sival_ptr = (void *) &timerId;

  for (i = 0; i < 100; i++)
    {
      printf ("cnt = %d\n", i);

      if (timer_create (CLOCK_REALTIME, &sigev, &timerId) < 0)
	{
	  perror ("timer_create");
	  continue;
	}

      res = timer_settime (timerId, 0, &itval, NULL);
      if (res < 0)
	perror ("timer_settime");

      res = timer_delete (timerId);
      if (res < 0)
	perror ("timer_delete");
    }

  return 0;
}

# define TEST_FUNCTION do_test ()
#else
# define TEST_FUNCTION 0
#endif

#include "../test-skeleton.c"
