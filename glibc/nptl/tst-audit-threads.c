/* Test multi-threading using LD_AUDIT.

   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* This test uses a dummy LD_AUDIT library (test-audit-threads-mod1) and a
   library with a huge number of functions in order to validate lazy symbol
   binding with an audit library.  We use one thread per CPU to test that
   concurrent lazy resolution does not have any defects which would cause
   the process to fail.  We use an LD_AUDIT library to force the testing of
   the relocation resolution caching code in the dynamic loader i.e.
   _dl_runtime_profile and _dl_profile_fixup.  */

#include <support/support.h>
#include <support/xthread.h>
#include <strings.h>
#include <stdlib.h>
#include <sys/sysinfo.h>

/* Declare the functions we are going to call.  */
#define externnum
#include "tst-audit-threads.h"
#undef externnum

int num_threads;
pthread_barrier_t barrier;

void
sync_all (int num)
{
  pthread_barrier_wait (&barrier);
}

void
call_all_ret_nums (void)
{
  /* Call each function one at a time from all threads.  */
#define callnum
#include "tst-audit-threads.h"
#undef callnum
}

void *
thread_main (void *unused)
{
  call_all_ret_nums ();
  return NULL;
}

#define STR2(X) #X
#define STR(X) STR2(X)

static int
do_test (void)
{
  int i;
  pthread_t *threads;

  num_threads = get_nprocs ();
  if (num_threads <= 1)
    num_threads = 2;

  /* Used to synchronize all the threads after calling each retNumN.  */
  xpthread_barrier_init (&barrier, NULL, num_threads);

  threads = (pthread_t *) xcalloc (num_threads, sizeof (pthread_t));
  for (i = 0; i < num_threads; i++)
    threads[i] = xpthread_create(NULL, thread_main, NULL);

  for (i = 0; i < num_threads; i++)
    xpthread_join(threads[i]);

  free (threads);

  return 0;
}

/* This test usually takes less than 3s to run.  However, there are cases that
   take up to 30s.  */
#define TIMEOUT 60
#include <support/test-driver.c>
