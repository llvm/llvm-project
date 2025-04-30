/* Test the interaction of fork and robust mutexes.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <support/check.h>
#include <support/test-driver.h>
#include <support/xthread.h>
#include <support/xunistd.h>
#include <sys/mman.h>

/* Data shared between processes. */
struct shared
{
  pthread_mutex_t parent_mutex;
  pthread_mutex_t child_mutex;
};

/* These flags control which mutex settings are enabled in the parent
   and child (separately).  */
enum mutex_bits
  {
    mutex_pshared = 1,
    mutex_robust = 2,
    mutex_pi = 4,
    mutex_check = 8,

    /* All bits combined.  */
    mutex_all_bits = 15,
  };

static void
mutex_init (pthread_mutex_t *mutex, int bits)
{
  pthread_mutexattr_t attr;
  xpthread_mutexattr_init (&attr);
  if (bits & mutex_pshared)
    xpthread_mutexattr_setpshared (&attr, PTHREAD_PROCESS_SHARED);
  if (bits & mutex_robust)
    xpthread_mutexattr_setrobust (&attr, PTHREAD_MUTEX_ROBUST);
  if (bits & mutex_pi)
    xpthread_mutexattr_setprotocol (&attr, PTHREAD_PRIO_INHERIT);
  if (bits & mutex_check)
    xpthread_mutexattr_settype (&attr, PTHREAD_MUTEX_ERRORCHECK);
  xpthread_mutex_init (mutex, &attr);
  xpthread_mutexattr_destroy (&attr);
}

static void
one_test (int parent_bits, int child_bits, int nonshared_bits,
          bool lock_nonshared, bool lock_child)
{

  struct shared *shared = xmmap (NULL, sizeof (*shared),
                                 PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_SHARED, -1);
  mutex_init (&shared->parent_mutex, parent_bits);
  mutex_init (&shared->child_mutex, child_bits);

  /* Acquire the parent mutex in the parent.  */
  xpthread_mutex_lock (&shared->parent_mutex);

  pthread_mutex_t nonshared_mutex;
  mutex_init (&nonshared_mutex, nonshared_bits);
  if (lock_nonshared)
    xpthread_mutex_lock (&nonshared_mutex);

  pid_t pid = xfork ();
  if (pid == 0)
    {
      /* Child process.  */
      if (lock_child)
        xpthread_mutex_lock (&shared->child_mutex);
      else
        xmunmap (shared, sizeof (*shared));
      if (lock_nonshared)
        /* Reinitialize the non-shared mutex if it was locked in the
           parent.  */
        mutex_init (&nonshared_mutex, nonshared_bits);
      xpthread_mutex_lock (&nonshared_mutex);
      /* For robust mutexes, the _exit call will perform the unlock
         instead.  */
      if (lock_child && !(child_bits & mutex_robust))
        xpthread_mutex_unlock (&shared->child_mutex);
      _exit (0);
    }
  /* Parent process. */
  {
    int status;
    xwaitpid (pid, &status, 0);
    TEST_VERIFY (status == 0);
  }

  if (parent_bits & mutex_check)
    /* Test for expected self-deadlock.  This is only possible to
       detect if the mutex is error-checking.  */
    TEST_VERIFY_EXIT (pthread_mutex_lock (&shared->parent_mutex) == EDEADLK);

  pid = xfork ();
  if (pid == 0)
    {
      /* Child process.  We can perform some checks only if we are
         dealing with process-shared mutexes.  */
      if (parent_bits & mutex_pshared)
        /* It must not be possible to acquire the parent mutex.

           NB: This check touches a mutex which has been acquired in
           the parent at fork time, so it might be deemed undefined
           behavior, pending the resolution of Austin Groups issue
           1112.  */
        TEST_VERIFY_EXIT (pthread_mutex_trylock (&shared->parent_mutex)
                          == EBUSY);
      if (lock_child && (child_bits & mutex_robust))
        {
          if (!(child_bits & mutex_pshared))
            /* No further tests possible.  */
            _exit (0);
          TEST_VERIFY_EXIT (pthread_mutex_lock (&shared->child_mutex)
                            == EOWNERDEAD);
          xpthread_mutex_consistent (&shared->child_mutex);
        }
      else
        /* We did not acquire the lock in the first child process, or
           we unlocked the mutex again because the mutex is not a
           robust mutex.  */
        xpthread_mutex_lock (&shared->child_mutex);
      xpthread_mutex_unlock (&shared->child_mutex);
      _exit (0);
    }
  /* Parent process. */
  {
    int status;
    xwaitpid (pid, &status, 0);
    TEST_VERIFY (status == 0);
  }

  if (lock_nonshared)
    xpthread_mutex_unlock (&nonshared_mutex);
  xpthread_mutex_unlock (&shared->parent_mutex);
  xpthread_mutex_destroy (&shared->parent_mutex);
  xpthread_mutex_destroy (&shared->child_mutex);
  xpthread_mutex_destroy (&nonshared_mutex);
  xmunmap (shared, sizeof (*shared));
}

static int
do_test (void)
{
  for (int parent_bits = 0; parent_bits <= mutex_all_bits; ++parent_bits)
    for (int child_bits = 0; child_bits <= mutex_all_bits; ++child_bits)
      for (int nonshared_bits = 0; nonshared_bits <= mutex_all_bits;
           ++nonshared_bits)
        for (int lock_nonshared = 0; lock_nonshared < 2; ++lock_nonshared)
          for (int lock_child = 0; lock_child < 2; ++lock_child)
            {
              if (test_verbose)
                printf ("info: parent_bits=0x%x child_bits=0x%x"
                        " nonshared_bits=0x%x%s%s\n",
                        parent_bits, child_bits, nonshared_bits,
                        lock_nonshared ? " lock_nonshared" : "",
                        lock_child ? " lock_child" : "");
              one_test (parent_bits, child_bits, nonshared_bits,
                        lock_nonshared, lock_child);
            }
  return 0;
}

#define TIMEOUT 100
#include <support/test-driver.c>
