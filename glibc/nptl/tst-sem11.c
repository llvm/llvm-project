#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <internaltypes.h>

#ifndef SEM_WAIT
# define SEM_WAIT(s) sem_wait (s)
#endif

static void *
tf (void *arg)
{
#ifdef PREPARE
  PREPARE
#endif
  SEM_WAIT (arg);
  return NULL;
}

int
main (void)
{
  int tries = 5;
  pthread_t th;
  union
  {
    sem_t s;
    struct new_sem ns;
  } u;
 again:
  if (sem_init (&u.s, 0, 0) != 0)
    {
      puts ("sem_init failed");
      return 1;
    }
#if __HAVE_64B_ATOMICS
  if ((u.ns.data >> SEM_NWAITERS_SHIFT) != 0)
#else
  if (u.ns.nwaiters != 0)
#endif
    {
      puts ("nwaiters not initialized");
      return 1;
    }

  if (pthread_create (&th, NULL, tf, &u.s) != 0)
    {
      puts ("pthread_create failed");
      return 1;
    }

  sleep (1);

  if (pthread_cancel (th) != 0)
    {
      puts ("pthread_cancel failed");
      return 1;
    }

  void *r;
  if (pthread_join (th, &r) != 0)
    {
      puts ("pthread_join failed");
      return 1;
    }
  if (r != PTHREAD_CANCELED && --tries > 0)
    {
      /* Maybe we get the scheduling right the next time.  */
      sem_destroy (&u.s);
      goto again;
    }

#if __HAVE_64B_ATOMICS
  if ((u.ns.data >> SEM_NWAITERS_SHIFT) != 0)
#else
  if (u.ns.nwaiters != 0)
#endif
    {
      puts ("nwaiters not reset");
      return 1;
    }

  return 0;
}
