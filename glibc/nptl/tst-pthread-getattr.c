/* Make sure that the stackaddr returned by pthread_getattr_np is
   reachable.

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

#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/param.h>
#include <pthread.h>
#include <alloca.h>
#include <assert.h>
#include <unistd.h>
#include <inttypes.h>

/* There is an obscure bug in the kernel due to which RLIMIT_STACK is sometimes
   returned as unlimited when it is not, which may cause this test to fail.
   There is also the other case where RLIMIT_STACK is intentionally set as
   unlimited or very high, which may result in a vma that is too large and again
   results in a test case failure.  To avoid these problems, we cap the stack
   size to one less than 8M.  See the following mailing list threads for more
   information about this problem:
   <https://sourceware.org/ml/libc-alpha/2012-06/msg00599.html>
   <https://sourceware.org/ml/libc-alpha/2012-06/msg00713.html>.  */
#define MAX_STACK_SIZE (8192 * 1024 - 1)

static size_t pagesize;

/* Test that the page in which TARGET lies is accessible.  This will
   segfault if the write fails.  This function has only half a page
   of thread stack left and so should not do anything and immediately
   return the address to which the stack reached.  */
static volatile uintptr_t
allocate_and_test (char *target)
{
  volatile char *mem = (char *) &mem;
  /* FIXME:  mem >= target for _STACK_GROWSUP.  */
  mem = alloca ((size_t) (mem - target));

  *mem = 42;
  return (uintptr_t) mem;
}

static int
get_self_pthread_attr (const char *id, void **stackaddr, size_t *stacksize)
{
  pthread_attr_t attr;
  int ret;
  pthread_t me = pthread_self ();

  if ((ret = pthread_getattr_np (me, &attr)) < 0)
    {
      printf ("%s: pthread_getattr_np failed: %s\n", id, strerror (ret));
      return 1;
    }

  if ((ret = pthread_attr_getstack (&attr, stackaddr, stacksize)) < 0)
    {
      printf ("%s: pthread_attr_getstack returned error: %s\n", id,
	      strerror (ret));
      return 1;
    }

  return 0;
}

/* Verify that the stack size returned by pthread_getattr_np is usable when
   the returned value is subject to rlimit.  */
static int
check_stack_top (void)
{
  struct rlimit stack_limit;
  void *stackaddr;
  size_t stacksize = 0;
  int ret;
  uintptr_t pagemask = ~(pagesize - 1);

  puts ("Verifying that stack top is accessible");

  ret = getrlimit (RLIMIT_STACK, &stack_limit);
  if (ret)
    {
      perror ("getrlimit failed");
      return 1;
    }

  printf ("current rlimit_stack is %zu\n", (size_t) stack_limit.rlim_cur);

  if (get_self_pthread_attr ("check_stack_top", &stackaddr, &stacksize))
    return 1;

  /* Reduce the rlimit to a page less that what is currently being returned
     (subject to a maximum of MAX_STACK_SIZE) so that we ensure that
     pthread_getattr_np uses rlimit.  The figure is intentionally unaligned so
     to verify that pthread_getattr_np returns an aligned stacksize that
     correctly fits into the rlimit.  We don't bother about the case where the
     stack is limited by the vma below it and not by the rlimit because the
     stacksize returned in that case is computed from the end of that vma and is
     hence safe.  */
  stack_limit.rlim_cur = MIN (stacksize - pagesize + 1, MAX_STACK_SIZE);
  printf ("Adjusting RLIMIT_STACK to %zu\n", (size_t) stack_limit.rlim_cur);
  if ((ret = setrlimit (RLIMIT_STACK, &stack_limit)) < 0)
    {
      perror ("setrlimit failed");
      return 1;
    }

  if (get_self_pthread_attr ("check_stack_top2", &stackaddr, &stacksize))
    return 1;

  printf ("Adjusted rlimit: stacksize=%zu, stackaddr=%p\n", stacksize,
          stackaddr);

  /* A lot of targets tend to write stuff on top of the user stack during
     context switches, so we cannot possibly safely go up to the very top of
     stack and test access there.  It is however sufficient to simply check if
     the top page is accessible, so we target our access halfway up the top
     page.  Thanks Chris Metcalf for this idea.  */
  uintptr_t mem = allocate_and_test (stackaddr + pagesize / 2);

  /* Before we celebrate, make sure we actually did test the same page.  */
  if (((uintptr_t) stackaddr & pagemask) != (mem & pagemask))
    {
      printf ("We successfully wrote into the wrong page.\n"
	      "Expected %#" PRIxPTR ", but got %#" PRIxPTR "\n",
	      (uintptr_t) stackaddr & pagemask, mem & pagemask);

      return 1;
    }

  puts ("Stack top tests done");

  return 0;
}

/* TODO: Similar check for thread stacks once the thread stack sizes are
   fixed.  */
static int
do_test (void)
{
  pagesize = sysconf (_SC_PAGESIZE);
  return check_stack_top ();
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
