/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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
#include <error.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>
#include <support/support.h>

#define N	4
#if __WORDSIZE == 64
#define GUARD_PATTERN 0xdeadbeafdeadbeaf
#else
#define GUARD_PATTERN 0xdeadbeaf
#endif

typedef struct {
       ucontext_t uctx;
       unsigned long	guard[3];
   } tst_context_t;

static char *stacks[N];
static size_t stack_size;
static tst_context_t ctx[N][2];
static volatile int failures;


static void
fct (long int n)
{
  char on_stack[1];

  /* Just to use the thread local descriptor.  */
  printf ("%ld: in %s now, on_stack = %p\n", n, __FUNCTION__, on_stack);
  errno = 0;

  if (ctx[n][1].uctx.uc_link != &ctx[n][0].uctx)
    {
      printf ("context[%ld][1] uc_link damaged, = %p\n", n,
	      ctx[n][1].uctx.uc_link);
      exit (1);
    }

  if ((ctx[n][0].guard[0] != GUARD_PATTERN)
  ||  (ctx[n][0].guard[1] != GUARD_PATTERN)
  ||  (ctx[n][0].guard[2] != GUARD_PATTERN))
    {
      printf ("%ld: %s context[0] overflow detected!\n", n, __FUNCTION__);
      ++failures;
    }

  if ((ctx[n][1].guard[0] != GUARD_PATTERN)
  ||  (ctx[n][1].guard[1] != GUARD_PATTERN)
  ||  (ctx[n][1].guard[2] != GUARD_PATTERN))
    {
      printf ("%ld: %s context[1] overflow detected!\n", n, __FUNCTION__);
      ++failures;
    }

  if (n < 0 || n >= N)
    {
      printf ("%ld out of range\n", n);
      exit (1);
    }

  if (on_stack < stacks[n] || on_stack >= stacks[n] + stack_size)
    {
      printf ("%ld: on_stack not on appropriate stack\n", n);
      exit (1);
    }
}


static void *
tf (void *arg)
{
  int n = (int) (long int) arg;

  ctx[n][0].guard[0] = GUARD_PATTERN;
  ctx[n][0].guard[1] = GUARD_PATTERN;
  ctx[n][0].guard[2] = GUARD_PATTERN;

  ctx[n][1].guard[0] = GUARD_PATTERN;
  ctx[n][1].guard[1] = GUARD_PATTERN;
  ctx[n][1].guard[2] = GUARD_PATTERN;

  if (getcontext (&ctx[n][1].uctx) != 0)
    {
      printf ("%d: cannot get context: %m\n", n);
      exit (1);
    }

  printf ("%d: %s: before makecontext\n", n, __FUNCTION__);

  ctx[n][1].uctx.uc_stack.ss_sp = stacks[n];
  ctx[n][1].uctx.uc_stack.ss_size = stack_size;
  ctx[n][1].uctx.uc_link = &ctx[n][0].uctx;
  makecontext (&ctx[n][1].uctx, (void (*) (void)) fct, 1, (long int) n);

  printf ("%d: %s: before swapcontext\n", n, __FUNCTION__);

  if (swapcontext (&ctx[n][0].uctx, &ctx[n][1].uctx) != 0)
    {
      ++failures;
      printf ("%d: %s: swapcontext failed\n", n, __FUNCTION__);
    }
  else
    printf ("%d: back in %s\n", n, __FUNCTION__);

  return NULL;
}


static volatile int global;


static int
do_test (void)
{
  int n;
  pthread_t th[N];
  ucontext_t mctx;

  stack_size = 2 * PTHREAD_STACK_MIN;
  for (int i = 0; i < N; i++)
    stacks[i] = xmalloc (stack_size);

  puts ("making contexts");
  if (getcontext (&mctx) != 0)
    {
      if (errno == ENOSYS)
	{
	  puts ("context handling not supported");
	  exit (0);
	}

      printf ("%s: getcontext: %m\n", __FUNCTION__);
      exit (1);
    }

  /* Play some tricks with this context.  */
  if (++global == 1)
    if (setcontext (&mctx) != 0)
      {
	puts ("setcontext failed");
	exit (1);
      }
  if (global != 2)
    {
      puts ("global not incremented twice");
      exit (1);
    }
  puts ("global OK");

  pthread_attr_t at;

  if (pthread_attr_init (&at) != 0)
    {
      puts ("attr_init failed");
      return 1;
    }

  if (pthread_attr_setstacksize (&at, 1 * 1024 * 1024) != 0)
    {
      puts ("attr_setstacksize failed");
      return 1;
    }

  for (n = 0; n < N; ++n)
    if (pthread_create (&th[n], &at, tf, (void *) (long int) n) != 0)
      {
	puts ("create failed");
	exit (1);
      }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  for (n = 0; n < N; ++n)
    if (pthread_join (th[n], NULL) != 0)
      {
	puts ("join failed");
	exit (1);
      }

  for (int i = 0; i < N; i++)
    free (stacks[i]);

  return failures;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
