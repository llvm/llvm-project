#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>


static pthread_mutex_t m;

static void *
tf (void *data)
{
  int err = pthread_mutex_lock (&m);
  if (err == EOWNERDEAD)
    {
      err = pthread_mutex_consistent (&m);
      if (err)
	{
	  puts ("pthread_mutex_consistent");
	  exit (1);
	}
    }
  else if (err)
    {
      puts ("pthread_mutex_lock");
      exit (1);
    }
  printf ("thread%ld got the lock.\n", (long int) data);
  sleep (1);
  /* exit without unlock */
  return NULL;
}

static int
do_test (void)
{
  int err, i;
  pthread_t t[3];
  pthread_mutexattr_t ma;

  pthread_mutexattr_init (&ma);
  err = pthread_mutexattr_setrobust (&ma, PTHREAD_MUTEX_ROBUST_NP);
  if (err)
    {
      puts ("pthread_mutexattr_setrobust");
      return 1;
    }
#ifdef ENABLE_PI
  if (pthread_mutexattr_setprotocol (&ma, PTHREAD_PRIO_INHERIT) != 0)
    {
      puts ("pthread_mutexattr_setprotocol failed");
      return 1;
    }
#endif
  err = pthread_mutex_init (&m, &ma);
#ifdef ENABLE_PI
  if (err == ENOTSUP)
    {
      puts ("PI robust mutexes not supported");
      return 0;
    }
#endif
  if (err)
    {
      puts ("pthread_mutex_init");
      return 1;
    }

  for (i = 0; i < sizeof (t) / sizeof (t[0]); i++)
    {
      err = pthread_create (&t[i], NULL, tf, (void *) (long int) i);
      if (err)
	{
	  puts ("pthread_create");
	  return 1;
	}
    }

  for (i = 0; i < sizeof (t) / sizeof (t[0]); i++)
    {
      err = pthread_join (t[i], NULL);
      if (err)
	{
	  puts ("pthread_join");
	  return 1;
	}
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
