/* Smoke test for malloc_info.
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

/* The purpose of this test is to provide a quick way to run
   malloc_info in a multi-threaded process.  */

#include <array_length.h>
#include <malloc.h>
#include <stdlib.h>
#include <support/support.h>
#include <support/xthread.h>

/* This barrier is used to have the main thread wait until the helper
   threads have performed their allocations.  */
static pthread_barrier_t barrier;

enum
  {
    /* Number of threads performing allocations.  */
    thread_count  = 4,

    /* Amount of memory allocation per thread.  This should be large
       enough to cause the allocation of multiple heaps per arena.  */
    per_thread_allocations
      = sizeof (void *) == 4 ? 16 * 1024 * 1024 : 128 * 1024 * 1024,
  };

static void *
allocation_thread_function (void *closure)
{
  struct list
  {
    struct list *next;
    long dummy[4];
  };

  struct list *head = NULL;
  size_t allocated = 0;
  while (allocated < per_thread_allocations)
    {
      struct list *new_head = xmalloc (sizeof (*new_head));
      allocated += sizeof (*new_head);
      new_head->next = head;
      head = new_head;
    }

  xpthread_barrier_wait (&barrier);

  /* Main thread prints first statistics here.  */

  xpthread_barrier_wait (&barrier);

  while (head != NULL)
    {
      struct list *next_head = head->next;
      free (head);
      head = next_head;
    }

  return NULL;
}

static int
do_test (void)
{
  xpthread_barrier_init (&barrier, NULL, thread_count + 1);

  pthread_t threads[thread_count];
  for (size_t i = 0; i < array_length (threads); ++i)
    threads[i] = xpthread_create (NULL, allocation_thread_function, NULL);

  xpthread_barrier_wait (&barrier);
  puts ("info: After allocation:");
  malloc_info (0, stdout);

  xpthread_barrier_wait (&barrier);
  for (size_t i = 0; i < array_length (threads); ++i)
    xpthread_join (threads[i]);

  puts ("\ninfo: After deallocation:");
  malloc_info (0, stdout);

  return 0;
}

#include <support/test-driver.c>
