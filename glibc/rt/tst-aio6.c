/* Test for timeout handling.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <aio.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>


#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  struct aiocb *arr[1];
  struct aiocb cb;
  char buf[100];
  struct timeval before;
  struct timeval after;
  struct timespec timeout;
  int fd[2];
  int result = 0;

  if (pipe (fd) != 0)
    {
      printf ("cannot create pipe: %m\n");
      return 1;
    }

  arr[0] = &cb;

  cb.aio_fildes = fd[0];
  cb.aio_lio_opcode = LIO_WRITE;
  cb.aio_reqprio = 0;
  cb.aio_buf = (void *) buf;
  cb.aio_nbytes = sizeof (buf) - 1;
  cb.aio_offset = 0;
  cb.aio_sigevent.sigev_notify = SIGEV_NONE;

  /* Try to read from stdin where nothing will be available.  */
  if (aio_read (arr[0]) < 0)
    {
      if (errno == ENOSYS)
	{
	  puts ("no aio support in this configuration");
	  return 0;
	}
      printf ("aio_read failed: %m\n");
      return 1;
    }

  /* Get the current time.  */
  gettimeofday (&before, NULL);

  /* Wait for input which is unsuccessful and therefore the function will
     time out.  */
  timeout.tv_sec = 3;
  timeout.tv_nsec = 0;
  if (aio_suspend ((const struct aiocb *const*) arr, 1, &timeout) != -1)
    {
      puts ("aio_suspend() didn't return -1");
      result = 1;
    }
  else if (errno != EAGAIN)
    {
      puts ("error not set to EAGAIN");
      result = 1;
    }
  else
    {
      gettimeofday (&after, NULL);
      if (after.tv_sec < before.tv_sec + 1)
	{
	  puts ("timeout came too early");
	  result = 1;
	}
    }

  return result;
}

#include "../test-skeleton.c"
