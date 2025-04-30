/* Test for notification mechanism in lio_listio.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>


static pthread_barrier_t b;


static void
thrfct (sigval_t arg)
{
  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("thread: barrier_wait failed");
      exit (1);
    }
}


static int
do_test (int argc, char *argv[])
{
  char name[] = "/tmp/aio2.XXXXXX";
  int fd;
  struct aiocb *arr[1];
  struct aiocb cb;
  static const char buf[] = "Hello World\n";

  fd = mkstemp (name);
  if (fd == -1)
    {
      printf ("cannot open temp name: %m\n");
      return 1;
    }

  unlink (name);

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  arr[0] = &cb;

  cb.aio_fildes = fd;
  cb.aio_lio_opcode = LIO_WRITE;
  cb.aio_reqprio = 0;
  cb.aio_buf = (void *) buf;
  cb.aio_nbytes = sizeof (buf) - 1;
  cb.aio_offset = 0;
  cb.aio_sigevent.sigev_notify = SIGEV_THREAD;
  cb.aio_sigevent.sigev_notify_function = thrfct;
  cb.aio_sigevent.sigev_notify_attributes = NULL;
  cb.aio_sigevent.sigev_value.sival_ptr = NULL;

  if (lio_listio (LIO_WAIT, arr, 1, NULL) < 0)
    {
      if (errno == ENOSYS)
	{
	  puts ("no aio support in this configuration");
	  return 0;
	}
      printf ("lio_listio failed: %m\n");
      return 1;
    }

  puts ("lio_listio returned");

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return 1;
    }

  puts ("all OK");

  return 0;
}

#include "../test-skeleton.c"
