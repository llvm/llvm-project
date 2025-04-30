/* Test allocation function behavior on allocation failure.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

/* This test case attempts to trigger various unusual conditions
   related to allocation failures, notably switching to a different
   arena, and falling back to mmap (via sysmalloc).  */

#include <errno.h>
#include <malloc.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

/* Wrapper for calloc with an optimization barrier.  */
static void *
__attribute__ ((noinline, noclone))
allocate_zeroed (size_t a, size_t b)
{
  return calloc (a, b);
}

/* System page size, as determined by sysconf (_SC_PAGE_SIZE).  */
static unsigned long page_size;

/* Test parameters. */
static size_t allocation_size;
static size_t alignment;
static enum {
  with_malloc,
  with_realloc,
  with_aligned_alloc,
  with_memalign,
  with_posix_memalign,
  with_valloc,
  with_pvalloc,
  with_calloc,
  last_allocation_function = with_calloc
} allocation_function;

/* True if an allocation function uses the alignment test
   parameter.  */
const static bool alignment_sensitive[last_allocation_function + 1] =
  {
    [with_aligned_alloc] = true,
    [with_memalign] = true,
    [with_posix_memalign] = true,
  };

/* Combined pointer/expected alignment result of an allocation
   function.  */
struct allocate_result {
  void *pointer;
  size_t alignment;
};

/* Call the allocation function specified by allocation_function, with
   allocation_size and alignment (if applicable) as arguments.  No
   alignment check.  */
static struct allocate_result
allocate_1 (void)
{
  switch (allocation_function)
    {
    case with_malloc:
      return (struct allocate_result)
        {malloc (allocation_size), _Alignof (max_align_t)};
    case with_realloc:
      {
        void *p = realloc (NULL, 16);
        void *q;
        if (p == NULL)
          q = NULL;
        else
          {
            q = realloc (p, allocation_size);
            if (q == NULL)
              free (p);
          }
        return (struct allocate_result) {q, _Alignof (max_align_t)};
      }
    case with_aligned_alloc:
      {
        void *p = aligned_alloc (alignment, allocation_size);
        return (struct allocate_result) {p, alignment};
      }
    case with_memalign:
      {
        void *p = memalign (alignment, allocation_size);
        return (struct allocate_result) {p, alignment};
      }
    case with_posix_memalign:
      {
        void *p;
        if (posix_memalign (&p, alignment, allocation_size))
          {
            if (errno == ENOMEM)
              p = NULL;
            else
              {
                printf ("error: posix_memalign (p, %zu, %zu): %m\n",
                        alignment, allocation_size);
                abort ();
              }
          }
        return (struct allocate_result) {p, alignment};
      }
    case with_valloc:
      {
        void *p = valloc (allocation_size);
        return (struct allocate_result) {p, page_size};
      }
    case with_pvalloc:
      {
        void *p = pvalloc (allocation_size);
        return (struct allocate_result) {p, page_size};
      }
    case with_calloc:
      {
        char *p = allocate_zeroed (1, allocation_size);
        /* Check for non-zero bytes.  */
        if (p != NULL)
          for (size_t i = 0; i < allocation_size; ++i)
            if (p[i] != 0)
              {
                printf ("error: non-zero byte at offset %zu\n", i);
                abort ();
              }
        return (struct allocate_result) {p, _Alignof (max_align_t)};
      }
    }
  abort ();
}

/* Call allocate_1 and perform the alignment check on the result.  */
static void *
allocate (void)
{
  struct allocate_result r = allocate_1 ();
  if ((((uintptr_t) r.pointer) & (r.alignment - 1)) != 0)
    {
      printf ("error: allocation function %d, size %zu not aligned to %zu\n",
              (int) allocation_function, allocation_size, r.alignment);
      abort ();
    }
  return r.pointer;
}

/* Barriers to synchronize thread creation and termination.  */
static pthread_barrier_t start_barrier;
static pthread_barrier_t end_barrier;

/* Thread function which performs the allocation test.  Called by
   pthread_create and from the main thread.  */
static void *
allocate_thread (void *closure)
{
  /* Wait for the creation of all threads.  */
  {
    int ret = pthread_barrier_wait (&start_barrier);
    if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD)
      {
        errno = ret;
        printf ("error: pthread_barrier_wait: %m\n");
        abort ();
      }
  }

  /* Allocate until we run out of memory, creating a single-linked
     list.  */
  struct list {
    struct list *next;
  };
  struct list *head = NULL;
  while (true)
    {
      struct list *e = allocate ();
      if (e == NULL)
        break;

      e->next = head;
      head = e;
    }

  /* Wait for the allocation of all available memory.  */
  {
    int ret = pthread_barrier_wait (&end_barrier);
    if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD)
      {
        errno = ret;
        printf ("error: pthread_barrier_wait: %m\n");
        abort ();
      }
  }

  /* Free the allocated memory.  */
  while (head != NULL)
    {
      struct list *next = head->next;
      free (head);
      head = next;
    }

  return NULL;
}

/* Number of threads (plus the main thread.  */
enum { thread_count = 8 };

/* Thread attribute to request creation of threads with a non-default
   stack size which is rather small.  This avoids interfering with the
   configured address space limit.  */
static pthread_attr_t small_stack;

/* Runs one test in multiple threads, all in a subprocess so that
   subsequent tests do not interfere with each other.  */
static void
run_one (void)
{
  /* Isolate the tests in a subprocess, so that we can start over
     from scratch.  */
  pid_t pid = fork ();
  if (pid == 0)
    {
      /* In the child process.  Create the allocation threads.  */
      pthread_t threads[thread_count];

      for (unsigned i = 0; i < thread_count; ++i)
        {
          int ret = pthread_create (threads + i, &small_stack, allocate_thread, NULL);
          if (ret != 0)
            {
              errno = ret;
              printf ("error: pthread_create: %m\n");
              abort ();
            }
        }

      /* Also run the test on the main thread.  */
      allocate_thread (NULL);

      for (unsigned i = 0; i < thread_count; ++i)
        {
          int ret = pthread_join (threads[i], NULL);
          if (ret != 0)
            {
              errno = ret;
              printf ("error: pthread_join: %m\n");
              abort ();
            }
        }
      _exit (0);
    }
  else if (pid < 0)
    {
      printf ("error: fork: %m\n");
      abort ();
    }

  /* In the parent process.  Wait for the child process to exit.  */
  int status;
  if (waitpid (pid, &status, 0) < 0)
    {
      printf ("error: waitpid: %m\n");
      abort ();
    }
  if (status != 0)
    {
      printf ("error: exit status %d from child process\n", status);
      exit (1);
    }
}

/* Run all applicable allocation functions for the current test
   parameters.  */
static void
run_allocation_functions (void)
{
  for (int af = 0; af <= last_allocation_function; ++af)
    {
      /* Run alignment-sensitive functions for non-default
         alignments.  */
      if (alignment_sensitive[af] != (alignment != 0))
        continue;
      allocation_function = af;
      run_one ();
    }
}

int
do_test (void)
{
  /* Limit the number of malloc arenas.  We use a very low number so
     that despute the address space limit configured below, all
     requested arenas a can be created.  */
  if (mallopt (M_ARENA_MAX, 2) == 0)
    {
      printf ("error: mallopt (M_ARENA_MAX) failed\n");
      return 1;
    }

  /* Determine the page size.  */
  {
    long ret = sysconf (_SC_PAGE_SIZE);
    if (ret < 0)
      {
        printf ("error: sysconf (_SC_PAGE_SIZE): %m\n");
        return 1;
      }
    page_size = ret;
  }

  /* Limit the size of the process, so that memory allocation in
     allocate_thread will eventually fail, without impacting the
     entire system.  */
  {
    struct rlimit limit;
    if (getrlimit (RLIMIT_AS, &limit) != 0)
      {
        printf ("getrlimit (RLIMIT_AS) failed: %m\n");
        return 1;
      }
    long target = 200 * 1024 * 1024;
    if (limit.rlim_cur == RLIM_INFINITY || limit.rlim_cur > target)
      {
        limit.rlim_cur = target;
        if (setrlimit (RLIMIT_AS, &limit) != 0)
          {
            printf ("setrlimit (RLIMIT_AS) failed: %m\n");
            return 1;
          }
      }
  }

  /* Initialize thread attribute with a reduced stack size.  */
  {
    int ret = pthread_attr_init (&small_stack);
    if (ret != 0)
      {
        errno = ret;
        printf ("error: pthread_attr_init: %m\n");
        abort ();
      }
    unsigned long stack_size = ((256 * 1024) / page_size) * page_size;
    if (stack_size < 4 * page_size)
      stack_size = 8 * page_size;
    ret = pthread_attr_setstacksize (&small_stack, stack_size);
    if (ret != 0)
      {
        errno = ret;
        printf ("error: pthread_attr_setstacksize: %m\n");
        abort ();
      }
  }

  /* Initialize the barriers.  We run thread_count threads, plus 1 for
     the main thread.  */
  {
    int ret = pthread_barrier_init (&start_barrier, NULL, thread_count + 1);
    if (ret != 0)
      {
        errno = ret;
        printf ("error: pthread_barrier_init: %m\n");
        abort ();
      }

    ret = pthread_barrier_init (&end_barrier, NULL, thread_count + 1);
    if (ret != 0)
      {
        errno = ret;
        printf ("error: pthread_barrier_init: %m\n");
        abort ();
      }
  }

  allocation_size = 144;
  run_allocation_functions ();
  allocation_size = page_size;
  run_allocation_functions ();

  alignment = 128;
  allocation_size = 512;
  run_allocation_functions ();

  allocation_size = page_size;
  run_allocation_functions ();

  allocation_size = 17 * page_size;
  run_allocation_functions ();

  /* Deallocation the barriers and the thread attribute.  */
  {
    int ret = pthread_barrier_destroy (&end_barrier);
    if (ret != 0)
      {
        errno = ret;
        printf ("error: pthread_barrier_destroy: %m\n");
        return 1;
      }
    ret = pthread_barrier_destroy (&start_barrier);
    if (ret != 0)
      {
        errno = ret;
        printf ("error: pthread_barrier_destroy: %m\n");
        return 1;
      }
    ret = pthread_attr_destroy (&small_stack);
    if (ret != 0)
      {
        errno = ret;
        printf ("error: pthread_attr_destroy: %m\n");
        return 1;
      }
  }

  return 0;
}

/* The repeated allocations take some time on slow machines.  */
#define TIMEOUT 100

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
