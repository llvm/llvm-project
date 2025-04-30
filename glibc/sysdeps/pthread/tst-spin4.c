#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

static int count = 0;

static void *
thread_add_one (void *arg)
{
  int tmp;
  pthread_spinlock_t *lock = (pthread_spinlock_t *) arg;

  /* When do_test holds the lock for 1 sec, the two thread will be
     in contention for the lock. */
  if (pthread_spin_lock (lock) != 0)
    {
      puts ("thread_add_one(): spin_lock failed");
      pthread_exit ((void *) 1l);
    }

  /* sleep 1s before modifying count */
  tmp = count;
  sleep (1);
  count = tmp + 1;

  if (pthread_spin_unlock (lock) != 0)
    {
      puts ("thread_add_one(): spin_unlock failed");
      pthread_exit ((void *) 1l);
    }

  return NULL;
}

static int
do_test (void)
{
  pthread_t thr1, thr2;
  pthread_spinlock_t lock;
  int tmp;

  if (pthread_spin_init (&lock, PTHREAD_PROCESS_PRIVATE) != 0)
    {
      puts ("spin_init failed");
      return 1;
    }

  if (pthread_spin_lock (&lock) != 0)
    {
      puts ("1st spin_lock failed");
      return 1;
    }

  if (pthread_create (&thr1, NULL, thread_add_one, (void *) &lock) != 0)
    {
      puts ("1st pthread_create failed");
      return 1;
    }

  if (pthread_create (&thr2, NULL, thread_add_one, (void *) &lock) != 0)
    {
      puts ("2nd pthread_create failed");
      return 1;
    }

  /* sleep 1s before modifying count */
  tmp = count;
  sleep (1);
  count = tmp + 1;

  if (pthread_spin_unlock (&lock) != 0)
    {
      puts ("1st spin_unlock failed");
      return 1;
    }

  void *status;
  if (pthread_join (thr1, &status) != 0)
    {
      puts ("1st pthread_join failed");
      return 1;
    }
  if (status != NULL)
    {
      puts ("failure in the 1st thread");
      return 1;
    }
  if (pthread_join (thr2, &status) != 0)
    {
      puts ("2nd pthread_join failed");
      return 1;
    }
  if (status != NULL)
    {
      puts ("failure in the 2nd thread");
      return 1;
    }

  if (count != 3)
    {
      printf ("count is %d, should be 3\n", count);
      return 1;
    }
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
