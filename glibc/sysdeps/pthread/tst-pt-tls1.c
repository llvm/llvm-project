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

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <support/support.h>
#include <support/check.h>
#include <support/xthread.h>

struct test_s
{
  __attribute__ ((aligned(0x20))) int a;
  __attribute__ ((aligned(0x200))) int b;
};

#define INIT_A 1
#define INIT_B 42
/* Deliberately not static.  */
__thread struct test_s s __attribute__ ((tls_model ("initial-exec"))) =
{
  .a = INIT_A,
  .b = INIT_B
};

/* Use noinline in combination with not static to ensure that the
   alignment check is really done.  Otherwise it was optimized out!  */
__attribute__ ((noinline)) void
check_alignment (const char *thr_name, const char *ptr_name,
		 int *ptr, int alignment)
{
  uintptr_t offset_aligment = ((uintptr_t) ptr) & (alignment - 1);
  if (offset_aligment)
    {
      FAIL_EXIT1 ("%s (%p) is not 0x%x-byte aligned in %s thread\n",
		  ptr_name, ptr, alignment, thr_name);
    }
}

static void
check_s (const char *thr_name)
{
  if (s.a != INIT_A || s.b != INIT_B)
    FAIL_EXIT1 ("initial value of s in %s thread wrong\n", thr_name);

  check_alignment (thr_name, "s.a", &s.a, 0x20);
  check_alignment (thr_name, "s.b", &s.b, 0x200);
}

static void *
tf (void *arg)
{
  check_s ("child");

  ++s.a;

  return NULL;
}


int
do_test (void)
{
  check_s ("main");

  pthread_attr_t a;

  xpthread_attr_init (&a);

#define STACK_SIZE (1 * 1024 * 1024)
  xpthread_attr_setstacksize (&a, STACK_SIZE);

#define N 10
  int i;
  for (i = 0; i < N; ++i)
    {
#define M 10
      pthread_t th[M];
      int j;
      for (j = 0; j < M; ++j, ++s.a)
	th[j] = xpthread_create (&a, tf, NULL);

      for (j = 0; j < M; ++j)
	xpthread_join (th[j]);
    }

  /* Also check the alignment of the tls variables if a misaligned stack is
     specified.  */
  pthread_t th;
  void *thr_stack = NULL;
  thr_stack = xposix_memalign (0x200, STACK_SIZE + 1);
  xpthread_attr_setstack (&a, thr_stack + 1, STACK_SIZE);
  th = xpthread_create (&a, tf, NULL);
  xpthread_join (th);
  free (thr_stack);

  xpthread_attr_destroy (&a);

  return 0;
}

#include <support/test-driver.c>
