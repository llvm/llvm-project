/* Tests for POSIX timer implementation using thread CPU clock.  */

#include <unistd.h>

#if _POSIX_THREADS && defined _POSIX_CPUTIME

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <pthread.h>

static clockid_t worker_thread_clock;

#define TEST_CLOCK worker_thread_clock
#define TEST_CLOCK_MISSING(clock) \
  (setup_test () ? "thread CPU clock timer support" : NULL)

/* This function is intended to rack up both user and system time.  */
static void *
chew_cpu (void *arg)
{
  while (1)
    {
      static volatile char buf[4096];
      for (int i = 0; i < 100; ++i)
	for (size_t j = 0; j < sizeof buf; ++j)
	  buf[j] = 0xaa;
      int nullfd = open ("/dev/null", O_WRONLY);
      for (int i = 0; i < 100; ++i)
	for (size_t j = 0; j < sizeof buf; ++j)
	  buf[j] = 0xbb;
      write (nullfd, (char *) buf, sizeof buf);
      close (nullfd);
    }

  return NULL;
}

static int
setup_test (void)
{
  /* Test timers on a thread CPU clock by having a worker thread eating
     CPU.  First make sure we can make such timers at all.  */

  pthread_t th;
  int e = pthread_create (&th, NULL, chew_cpu, NULL);
  if (e != 0)
    {
      printf ("pthread_create: %s\n", strerror (e));
      exit (1);
    }

  e = pthread_getcpuclockid (th, &worker_thread_clock);
  if (e == EPERM || e == ENOENT || e == ENOTSUP)
    {
      puts ("pthread_getcpuclockid does not support other threads");
      return 1;
    }
  if (e != 0)
    {
      printf ("pthread_getcpuclockid: %s\n", strerror (e));
      exit (1);
    }

  timer_t t;
  if (timer_create (TEST_CLOCK, NULL, &t) != 0)
    {
      printf ("timer_create: %m\n");
      return 1;
    }
  timer_delete (t);

  return 0;
}

#else
# define TEST_CLOCK_MISSING(clock) "process clocks"
#endif

#include "tst-timer4.c"
