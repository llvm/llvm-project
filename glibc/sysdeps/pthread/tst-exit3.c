#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int do_test (void);

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static pthread_barrier_t b;


static void *
tf2 (void *arg)
{
  while (1)
    sleep (100);

  /* NOTREACHED */
  return NULL;
}


static void *
tf (void *arg)
{
  pthread_t th;

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  e = pthread_create (&th, NULL, tf2, NULL);
  if (e != 0)
    {
      printf ("create failed: %s\n", strerror (e));
      exit (1);
    }

  /* Terminate only this thread.  */
  return NULL;
}


static int
do_test (void)
{
  pthread_t th;

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  int e = pthread_create (&th, NULL, tf, NULL);
  if (e != 0)
    {
      printf ("create failed: %s\n", strerror (e));
      exit (1);
    }

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  delayed_exit (3);

  /* Terminate only this thread.  */
  pthread_exit (NULL);

  /* NOTREACHED */
  return 1;
}
