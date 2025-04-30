/* Test for bogus per-thread deletion of timers.  */

#include <stdio.h>
#include <error.h>
#include <time.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#if _POSIX_THREADS
# include <pthread.h>


/* Creating timers in another thread should work too.  */
static void *
do_timer_create (void *arg)
{
  struct sigevent *const sigev = arg;
  timer_t *const timerId = sigev->sigev_value.sival_ptr;
  if (timer_create (CLOCK_REALTIME, sigev, timerId) < 0)
    {
      printf ("timer_create: %m\n");
      return NULL;
    }
  return timerId;
}


static int
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

  sigev.sigev_notify = SIGEV_SIGNAL;
  sigev.sigev_signo = SIGALRM;
  sigev.sigev_value.sival_ptr = (void *) &timerId;

  for (i = 0; i < 100; i++)
    {
      printf ("cnt = %d\n", i);

      pthread_t thr;
      res = pthread_create (&thr, NULL, &do_timer_create, &sigev);
      if (res)
	{
	  printf ("pthread_create: %s\n", strerror (res));
	  continue;
	}
      void *val;
      res = pthread_join (thr, &val);
      if (res)
	{
	  printf ("pthread_join: %s\n", strerror (res));
	  continue;
	}
      if (val == NULL)
	continue;

      res = timer_settime (timerId, 0, &itval, NULL);
      if (res < 0)
	printf ("timer_settime: %m\n");

      res = timer_delete (timerId);
      if (res < 0)
	printf ("timer_delete: %m\n");
    }

  return 0;
}

# define TEST_FUNCTION do_test ()
#else
# define TEST_FUNCTION 0
#endif

#include "../test-skeleton.c"
