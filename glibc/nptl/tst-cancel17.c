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

#include <aio.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


static pthread_barrier_t b;


/* Cleanup handling test.  */
static int cl_called;

static void
cl (void *arg)
{
  ++cl_called;
}


static void *
tf (void *arg)
{
  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("tf: barrier_wait failed");
      exit (1);
    }

  pthread_cleanup_push (cl, NULL);

  const struct aiocb *l[1] = { arg };

  TEMP_FAILURE_RETRY (aio_suspend (l, 1, NULL));

  pthread_cleanup_pop (0);

  puts ("tf: aio_suspend returned");

  exit (1);
}


static void *
tf2 (void *arg)
{
  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("tf2: barrier_wait failed");
      exit (1);
    }

  pthread_cleanup_push (cl, NULL);

  const struct aiocb *l[1] = { arg };
  struct timespec ts = { .tv_sec = 1000, .tv_nsec = 0 };

  TEMP_FAILURE_RETRY (aio_suspend (l, 1, &ts));

  pthread_cleanup_pop (0);

  puts ("tf2: aio_suspend returned");

  exit (1);
}


static int
do_test (void)
{
  int fds[2];
  if (pipe (fds) != 0)
    {
      puts ("pipe failed");
      return 1;
    }

  struct aiocb a, a2, *ap;
  char mem[1];
  memset (&a, '\0', sizeof (a));
  a.aio_fildes = fds[0];
  a.aio_buf = mem;
  a.aio_nbytes = sizeof (mem);
  if (aio_read (&a) != 0)
    {
      puts ("aio_read failed");
      return 1;
    }

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, &a) != 0)
    {
      puts ("1st create failed");
      return 1;
    }

  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  struct timespec  ts = { .tv_sec = 0, .tv_nsec = 100000000 };
  while (nanosleep (&ts, &ts) != 0)
    continue;

  puts ("going to cancel tf in-time");
  if (pthread_cancel (th) != 0)
    {
      puts ("1st cancel failed");
      return 1;
    }

  void *status;
  if (pthread_join (th, &status) != 0)
    {
      puts ("1st join failed");
      return 1;
    }
  if (status != PTHREAD_CANCELED)
    {
      puts ("1st thread not canceled");
      return 1;
    }

  if (cl_called == 0)
    {
      puts ("tf cleanup handler not called");
      return 1;
    }
  if (cl_called > 1)
    {
      puts ("tf cleanup handler called more than once");
      return 1;
    }

  cl_called = 0;

  if (pthread_create (&th, NULL, tf2, &a) != 0)
    {
      puts ("2nd create failed");
      return 1;
    }

  r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("2nd barrier_wait failed");
      exit (1);
    }

  ts.tv_sec = 0;
  ts.tv_nsec = 100000000;
  while (nanosleep (&ts, &ts) != 0)
    continue;

  puts ("going to cancel tf2 in-time");
  if (pthread_cancel (th) != 0)
    {
      puts ("2nd cancel failed");
      return 1;
    }

  if (pthread_join (th, &status) != 0)
    {
      puts ("2nd join failed");
      return 1;
    }
  if (status != PTHREAD_CANCELED)
    {
      puts ("2nd thread not canceled");
      return 1;
    }

  if (cl_called == 0)
    {
      puts ("tf2 cleanup handler not called");
      return 1;
    }
  if (cl_called > 1)
    {
      puts ("tf2 cleanup handler called more than once");
      return 1;
    }

  puts ("in-time cancellation succeeded");

  ap = &a;
  if (aio_cancel (fds[0], &a) != AIO_CANCELED)
    {
      puts ("aio_cancel failed");
      /* If aio_cancel failed, we cannot reuse aiocb a.  */
      ap = &a2;
    }


  cl_called = 0;

  size_t len2 = fpathconf (fds[1], _PC_PIPE_BUF);
  size_t page_size = sysconf (_SC_PAGESIZE);
  len2 = 20 * (len2 < page_size ? page_size : len2) + sizeof (mem) + 1;
  char *mem2 = malloc (len2);
  if (mem2 == NULL)
    {
      puts ("could not allocate memory for pipe write");
      return 1;
    }

  memset (ap, '\0', sizeof (*ap));
  ap->aio_fildes = fds[1];
  ap->aio_buf = mem2;
  ap->aio_nbytes = len2;
  if (aio_write (ap) != 0)
    {
      puts ("aio_write failed");
      return 1;
    }

  if (pthread_create (&th, NULL, tf, ap) != 0)
    {
      puts ("3rd create failed");
      return 1;
    }

  puts ("going to cancel tf early");
  if (pthread_cancel (th) != 0)
    {
      puts ("3rd cancel failed");
      return 1;
    }

  r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("3rd barrier_wait failed");
      exit (1);
    }

  if (pthread_join (th, &status) != 0)
    {
      puts ("3rd join failed");
      return 1;
    }
  if (status != PTHREAD_CANCELED)
    {
      puts ("3rd thread not canceled");
      return 1;
    }

  if (cl_called == 0)
    {
      puts ("tf cleanup handler not called");
      return 1;
    }
  if (cl_called > 1)
    {
      puts ("tf cleanup handler called more than once");
      return 1;
    }

  cl_called = 0;

  if (pthread_create (&th, NULL, tf2, ap) != 0)
    {
      puts ("4th create failed");
      return 1;
    }

  puts ("going to cancel tf2 early");
  if (pthread_cancel (th) != 0)
    {
      puts ("4th cancel failed");
      return 1;
    }

  r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("4th barrier_wait failed");
      exit (1);
    }

  if (pthread_join (th, &status) != 0)
    {
      puts ("4th join failed");
      return 1;
    }
  if (status != PTHREAD_CANCELED)
    {
      puts ("4th thread not canceled");
      return 1;
    }

  if (cl_called == 0)
    {
      puts ("tf2 cleanup handler not called");
      return 1;
    }
  if (cl_called > 1)
    {
      puts ("tf2 cleanup handler called more than once");
      return 1;
    }

  puts ("early cancellation succeeded");

  if (ap == &a2)
    {
      /* The aio_read(&a) was not canceled because the read request was
	 already in progress. In the meanwhile aio_write(ap) wrote something
	 to the pipe and the read request either has already been finished or
	 is able to read the requested byte.
	 Wait for the read request before returning from this function because
	 the return value and error code from the read syscall will be written
	 to the struct aiocb a, which lies on the stack of this function.
	 Otherwise the stack from subsequent function calls - e.g. _dl_fini -
	 will be corrupted, which can lead to undefined behaviour like a
	 segmentation fault.  */
      const struct aiocb *l[1] = { &a };
      TEMP_FAILURE_RETRY (aio_suspend(l, 1, NULL));
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
