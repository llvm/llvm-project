/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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
#include <shlib-compat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* LinuxThreads pthread_cleanup_{push,pop} helpers.  */
extern void _pthread_cleanup_push (struct _pthread_cleanup_buffer *__buffer,
                                   void (*__routine) (void *),
                                   void *__arg);
compat_symbol_reference (libpthread, _pthread_cleanup_push,
                         _pthread_cleanup_push, GLIBC_2_0);
extern void _pthread_cleanup_pop (struct _pthread_cleanup_buffer *__buffer,
                                  int __execute);
compat_symbol_reference (libpthread, _pthread_cleanup_pop,
                         _pthread_cleanup_pop, GLIBC_2_0);

static int fds[2];
static pthread_barrier_t b2;
static int global;

/* Defined in tst-cleanup4aux.c, never compiled with -fexceptions.  */
extern void fn5 (void);
extern void fn7 (void);
extern void fn9 (void);

void
clh (void *arg)
{
  int val = (long int) arg;

  printf ("clh (%d)\n", val);

  global *= val;
  global += val;
}


static __attribute__((noinline)) void
fn_read (void)
{
  int r = pthread_barrier_wait (&b2);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      printf ("%s: barrier_wait failed\n", __FUNCTION__);
      exit (1);
    }

  char c;
  read (fds[0], &c, 1);
}


__attribute__((noinline)) void
fn0 (void)
{
  pthread_cleanup_push (clh, (void *) 1l);

  fn_read ();

  pthread_cleanup_pop (1);
}


__attribute__((noinline)) void
fn1 (void)
{
  /* This is the old LinuxThreads pthread_cleanup_{push,pop}.  */
  struct _pthread_cleanup_buffer b;
  _pthread_cleanup_push (&b, clh, (void *) 2l);

  fn0 ();

  _pthread_cleanup_pop (&b, 1);
}


static __attribute__((noinline)) void
fn2 (void)
{
  pthread_cleanup_push (clh, (void *) 3l);

  fn1 ();

  pthread_cleanup_pop (1);
}


static void *
tf (void *a)
{
  switch ((long) a)
    {
    case 0:
      fn2 ();
      break;
    case 1:
      fn5 ();
      break;
    case 2:
      fn7 ();
      break;
    case 3:
      fn9 ();
      break;
    }

  return NULL;
}


int
do_test (void)
{
  int result = 0;

  if (pipe (fds) != 0)
    {
      puts ("pipe failed");
      exit (1);
    }

  if (pthread_barrier_init (&b2, NULL, 2) != 0)
    {
      puts ("b2 init failed");
      exit (1);
    }

  const int expect[] =
    {
      15,	/* 1 2 3 */
      276,	/* 1 4 5 6 */
      120,	/* 1 7 8 */
      460	/* 1 2 9 10 */
    };

  long i;
  for (i = 0; i < 4; ++i)
    {
      global = 0;

      printf ("test %ld\n", i);

      pthread_t th;
      if (pthread_create (&th, NULL, tf, (void *) i) != 0)
	{
	  puts ("create failed");
	  exit (1);
	}

      int e = pthread_barrier_wait (&b2);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("%s: barrier_wait failed\n", __FUNCTION__);
	  exit (1);
	}

      pthread_cancel (th);

      void *r;
      if ((e = pthread_join (th, &r)) != 0)
	{
	  printf ("join failed: %d\n", e);
	  _exit (1);
	}

      if (r != PTHREAD_CANCELED)
	{
	  puts ("thread not canceled");
	  exit (1);
	}

      if (global != expect[i])
	{
	  printf ("global = %d, expected %d\n", global, expect[i]);
	  result = 1;
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
