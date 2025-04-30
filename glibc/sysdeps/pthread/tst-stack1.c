/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include <unistd.h>


static void *stack;
static size_t size;


static void *
tf (void *a)
{
  int result = 0;

  puts ("child start");

  pthread_attr_t attr;
  if (pthread_getattr_np (pthread_self (), &attr) != 0)
    {
      puts ("getattr_np failed");
      exit (1);
    }

  size_t test_size;
  void *test_stack;
  if (pthread_attr_getstack (&attr, &test_stack, &test_size) != 0)
    {
      puts ("attr_getstack failed");
      exit (1);
    }

  if (test_size != size)
    {
      printf ("child: reported size differs: is %zu, expected %zu\n",
	      test_size, size);
      result = 1;
    }

  if (test_stack != stack)
    {
      printf ("child: reported stack address differs: is %p, expected %p\n",
	      test_stack, stack);
      result = 1;
    }

  puts ("child OK");

  return result ? (void *) 1l : NULL;
}


int
do_test (void)
{
  int result = 0;

  size = 4 * getpagesize ();
#ifdef PTHREAD_STACK_MIN
  size = MAX (size, PTHREAD_STACK_MIN);
#endif
  if (posix_memalign (&stack, getpagesize (), size) != 0)
    {
      puts ("out of memory while allocating the stack memory");
      exit (1);
    }

  pthread_attr_t attr;
  if (pthread_attr_init (&attr) != 0)
    {
      puts ("attr_init failed");
      exit (1);
    }

  puts ("attr_setstack");
  if (pthread_attr_setstack (&attr, stack, size) != 0)
    {
      puts ("attr_setstack failed");
      exit (1);
    }

  size_t test_size;
  void *test_stack;
  puts ("attr_getstack");
  if (pthread_attr_getstack (&attr, &test_stack, &test_size) != 0)
    {
      puts ("attr_getstack failed");
      exit (1);
    }

  if (test_size != size)
    {
      printf ("reported size differs: is %zu, expected %zu\n",
	      test_size, size);
      result = 1;
    }

  if (test_stack != stack)
    {
      printf ("reported stack address differs: is %p, expected %p\n",
	      test_stack, stack);
      result = 1;
    }

  puts ("create");

  pthread_t th;
  if (pthread_create (&th, &attr, tf, NULL) != 0)
    {
      puts ("failed to create thread");
      exit (1);
    }

  void *status;
  if (pthread_join (th, &status) != 0)
    {
      puts ("join failed");
      exit (1);
    }

  result |= status != NULL;

  return result;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
