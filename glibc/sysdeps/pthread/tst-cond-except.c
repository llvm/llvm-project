/* Verify that exception table for pthread_cond_wait is correct.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

pthread_mutex_t mutex;
pthread_cond_t cond;

#define CHECK_RETURN_VAL_OR_FAIL(ret,str) \
  ({ if ((ret) != 0) \
       { \
         printf ("%s failed: %s\n", (str), strerror (ret)); \
         ret = 1; \
         goto out; \
       } \
  })


void
clean (void *arg)
{
  puts ("clean: Unlocking mutex...");
  pthread_mutex_unlock ((pthread_mutex_t *) arg);
  puts ("clean: Mutex unlocked...");
}

void *
thr (void *arg)
{
  int ret = 0;
  pthread_mutexattr_t mutexAttr;
  ret = pthread_mutexattr_init (&mutexAttr);
  CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_mutexattr_init");

  ret = pthread_mutexattr_setprotocol (&mutexAttr, PTHREAD_PRIO_INHERIT);
  CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_mutexattr_setprotocol");

  ret = pthread_mutex_init (&mutex, &mutexAttr);
  CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_mutex_init");

  ret = pthread_cond_init (&cond, 0);
  CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_cond_init");

  puts ("th: Init done, entering wait...");

  pthread_cleanup_push (clean, (void *) &mutex);
  ret = pthread_mutex_lock (&mutex);
  CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_mutex_lock");
  while (1)
    {
      ret = pthread_cond_wait (&cond, &mutex);
      CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_cond_wait");
    }
  pthread_cleanup_pop (1);

out:
  return (void *) (uintptr_t) ret;
}

int
do_test (void)
{
  pthread_t thread;
  int ret = 0;
  void *thr_ret = 0;
  ret = pthread_create (&thread, 0, thr, &thr_ret);
  CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_create");

  puts ("main: Thread created, waiting a bit...");
  sleep (2);

  puts ("main: Cancelling thread...");
  ret = pthread_cancel (thread);
  CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_cancel");

  puts ("main: Joining th...");
  ret = pthread_join (thread, NULL);
  CHECK_RETURN_VAL_OR_FAIL (ret, "pthread_join");

  if (thr_ret != NULL)
    return 1;

  puts ("main: Joined thread, done!");

out:
  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
