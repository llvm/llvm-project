#include <errno.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <internaltypes.h>
#include <support/check.h>

/* A bogus clock value that tells run_test to use sem_timedwait rather than
   sem_clockwait.  */
#define CLOCK_USE_TIMEDWAIT (-1)

typedef int (*waitfn_t)(sem_t *, struct timespec *);

static void
do_test_wait (waitfn_t waitfn, const char *fnname)
{
  union
  {
    sem_t s;
    struct new_sem ns;
  } u;

  printf ("do_test_wait: %s\n", fnname);

  TEST_COMPARE (sem_init (&u.s, 0, 0), 0);

  struct timespec ts = { 0, 1000000001 };	/* Invalid.  */
  errno = 0;
  TEST_VERIFY_EXIT (waitfn (&u.s, &ts) < 0);
  TEST_COMPARE (errno, EINVAL);

#if __HAVE_64B_ATOMICS
  unsigned int nwaiters = (u.ns.data >> SEM_NWAITERS_SHIFT);
#else
  unsigned int nwaiters = u.ns.nwaiters;
#endif
  TEST_COMPARE (nwaiters, 0);

  ts.tv_sec = /* Invalid.  */ -2;
  ts.tv_nsec = 0;
  errno = 0;
  TEST_VERIFY_EXIT (waitfn (&u.s, &ts) < 0);
  TEST_COMPARE (errno, ETIMEDOUT);
#if __HAVE_64B_ATOMICS
  nwaiters = (u.ns.data >> SEM_NWAITERS_SHIFT);
#else
  nwaiters = u.ns.nwaiters;
#endif
  TEST_COMPARE (nwaiters, 0);
}

int test_sem_timedwait (sem_t *sem, struct timespec *ts)
{
  return sem_timedwait (sem, ts);
}

int test_sem_clockwait_monotonic (sem_t *sem, struct timespec *ts)
{
  return sem_clockwait (sem, CLOCK_MONOTONIC, ts);
}

int test_sem_clockwait_realtime (sem_t *sem, struct timespec *ts)
{
  return sem_clockwait (sem, CLOCK_REALTIME, ts);
}

static int do_test (void)
{
  do_test_wait (&test_sem_timedwait,
                "sem_timedwait");
  do_test_wait (&test_sem_clockwait_monotonic,
                "sem_clockwait(monotonic)");
  do_test_wait (&test_sem_clockwait_realtime,
                "sem_clockwait(realtime)");
  return 0;
}

#include <support/test-driver.c>
