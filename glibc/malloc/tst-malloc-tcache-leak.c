/* Bug 22111: Test that threads do not leak their per thread cache.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* The point of this test is to start and exit a large number of
   threads, while at the same time looking to see if the used
   memory grows with each round of threads run.  If the memory
   grows above some linear bound we declare the test failed and
   that the malloc implementation is leaking memory with each
   thread.  This is a good indicator that the thread local cache
   is leaking chunks.  */

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <pthread.h>
#include <assert.h>
#include <libc-diag.h>

#include <support/check.h>
#include <support/support.h>
#include <support/xthread.h>

void *
worker (void *data)
{
  void *ret;
  /* Allocate an arbitrary amount of memory that is known to fit into
     the thread local cache (tcache).  If we have at least 64 bins
     (default e.g. TCACHE_MAX_BINS) we should be able to allocate 32
     bytes and force malloc to fill the tcache.  We are assuming tcahce
     init happens at the first small alloc, but it might in the future
     be deferred to some other point.  Therefore to future proof this
     test we include a full alloc/free/alloc cycle for the thread.  We
     need a compiler barrier to avoid the removal of the useless
     alloc/free.  We send some memory back to main to have the memory
     freed after the thread dies, as just another check that the chunks
     that were previously in the tcache are still OK to free after
     thread death.  */
  ret = xmalloc (32);
  __asm__ volatile ("" ::: "memory");
  free (ret);
  return (void *) xmalloc (32);
}

static int
do_test (void)
{
  pthread_t *thread;
  struct mallinfo info_before, info_after;
  void *retval;

  /* This is an arbitrary choice. We choose a total of THREADS
     threads created and joined.  This gives us enough iterations to
     show a leak.  */
  int threads = 100000;

  /* Avoid there being 0 malloc'd data at this point by allocating the
     pthread_t required to run the test.  */
  thread = (pthread_t *) xcalloc (1, sizeof (pthread_t));

  /* The test below covers the deprecated mallinfo function.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdeprecated-declarations");

  info_before = mallinfo ();

  assert (info_before.uordblks != 0);

  printf ("INFO: %d (bytes) are in use before starting threads.\n",
          info_before.uordblks);

  for (int loop = 0; loop < threads; loop++)
    {
      *thread = xpthread_create (NULL, worker, NULL);
      retval = xpthread_join (*thread);
      free (retval);
    }

  info_after = mallinfo ();
  printf ("INFO: %d (bytes) are in use after all threads joined.\n",
          info_after.uordblks);

  /* We need to compare the memory in use before and the memory in use
     after starting and joining THREADS threads.  We almost always grow
     memory slightly, but not much. Consider that if even 1-byte leaked
     per thread we'd have THREADS bytes of additional memory, and in
     general the in-use at the start of main is quite low.  We will
     always leak a full malloc chunk, and never just 1-byte, therefore
     anything above "+ threads" from the start (constant offset) is a
     leak.  Obviously this assumes no thread-related malloc'd internal
     libc data structures persist beyond the thread death, and any that
     did would limit the number of times you could call pthread_create,
     which is a QoI we'd want to detect and fix.  */
  if (info_after.uordblks > (info_before.uordblks + threads))
    FAIL_EXIT1 ("Memory usage after threads is too high.\n");

  DIAG_POP_NEEDS_COMMENT;

  /* Did not detect excessive memory usage.  */
  free (thread);
  exit (0);
}

#define TIMEOUT 50
#include <support/test-driver.c>
