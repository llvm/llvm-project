#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <internal-signals.h>


static pthread_barrier_t b;
static pthread_t th2;


static void *
tf2 (void *arg)
{
  sigset_t mask;
  if (pthread_sigmask (SIG_SETMASK, NULL, &mask) != 0)
    {
      puts ("pthread_sigmask failed");
      exit (1);
    }
#ifdef SIGCANCEL
  if (sigismember (&mask, SIGCANCEL))
    {
      puts ("SIGCANCEL blocked in new thread");
      exit (1);
    }
#endif

  /* Sync with the main thread so that we do not test anything else.  */
  int e = pthread_barrier_wait (&b);
  if (e != 0  && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  while (1)
    {
      /* Just a cancelable call.  */
      struct timespec ts = { 10000, 0 };
      nanosleep (&ts, 0);
    }

  return NULL;
}


static void
unwhand (void *arg)
{
  if (pthread_create (&th2, NULL, tf2, NULL) != 0)
    {
      puts ("unwhand: create failed");
      exit (1);
    }
}


static void *
tf (void *arg)
{
  pthread_cleanup_push (unwhand, NULL);

  /* Sync with the main thread so that we do not test anything else.  */
  int e = pthread_barrier_wait (&b);
  if (e != 0  && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  while (1)
    {
      /* Just a cancelable call.  */
      struct timespec ts = { 10000, 0 };
      nanosleep (&ts, 0);
    }

  pthread_cleanup_pop (0);

  return NULL;
}


static int
do_test (void)
{
  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  pthread_t th1;
  if (pthread_create (&th1, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0  && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return 1;
    }

  /* Make sure tf1 enters nanosleep.  */
  struct timespec ts = { 0, 500000000 };
  while (nanosleep (&ts, &ts) != 0)
    ;

  if (pthread_cancel (th1) != 0)
    {
      puts ("1st cancel failed");
      return 1;
    }

  void *res;
  if (pthread_join (th1, &res) != 0)
    {
      puts ("1st join failed");
      return 1;
    }
  if (res != PTHREAD_CANCELED)
    {
      puts ("1st thread not canceled");
      return 1;
    }

  e = pthread_barrier_wait (&b);
  if (e != 0  && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return 1;
    }

  /* Make sure tf2 enters nanosleep.  */
  ts.tv_sec = 0;
  ts.tv_nsec = 500000000;
  while (nanosleep (&ts, &ts) != 0)
    ;

  puts ("calling pthread_cancel the second time");
  if (pthread_cancel (th2) != 0)
    {
      puts ("2nd cancel failed");
      return 1;
    }

  puts ("calling pthread_join the second time");
  if (pthread_join (th2, &res) != 0)
    {
      puts ("2nd join failed");
      return 1;
    }
  if (res != PTHREAD_CANCELED)
    {
      puts ("2nd thread not canceled");
      return 1;
    }

  if (pthread_barrier_destroy (&b) != 0)
    {
      puts ("barrier_destroy failed");
      return 0;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
