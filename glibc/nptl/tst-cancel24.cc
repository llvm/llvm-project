#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


static volatile bool destr_called;
static volatile bool except_caught;

static pthread_barrier_t b;


struct monitor
{
  // gcc is broken and would generate a warning without this dummy
  // constructor.
  monitor () { }
  ~monitor() { destr_called = true; }
};


static void *
tf (void *arg)
{
  sem_t *s = static_cast<sem_t *> (arg);

  try
    {
      monitor m;

      pthread_barrier_wait (&b);

      while (1)
      sem_wait (s);
    }
  catch (...)
    {
      except_caught = true;
      throw;
    }

  return NULL;
}


static int
do_test ()
{
  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  sem_t s;
  if (sem_init (&s, 0, 0) != 0)
    {
      puts ("sem_init failed");
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, &s) != 0)
    {
      puts ("pthread_create failed");
      return 1;
    }

  pthread_barrier_wait (&b);

  /* There is unfortunately no better method to try to assure the
     child thread reached the sem_wait call and is actually waiting
     than to sleep here.  */
  sleep (1);

  if (pthread_cancel (th) != 0)
    {
      puts ("cancel failed");
      return 1;
    }

  void *res;
  if (pthread_join (th, &res) != 0)
    {
      puts ("join failed");
      return 1;
    }

  if (res != PTHREAD_CANCELED)
    {
      puts ("thread was not canceled");
      return 1;
    }

  if (! except_caught)
    {
      puts ("exception not caught");
      return 1;
    }

  if (! destr_called)
    {
      puts ("destructor not called");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#define TIMEOUT 3
#include "../test-skeleton.c"
