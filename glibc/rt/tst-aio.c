/* Tests for AIO in librt.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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
#include <error.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>


/* Prototype for our test function.  */
extern void do_prepare (int argc, char *argv[]);
extern int do_test (int argc, char *argv[]);

/* We have a preparation function.  */
#define PREPARE do_prepare

/* This defines the `main' function and some more.  */
#include <test-skeleton.c>


/* These are for the temporary file we generate.  */
char *name;
int fd;

void
do_prepare (int argc, char *argv[])
{
  size_t name_len;

  name_len = strlen (test_dir);
  name = xmalloc (name_len + sizeof ("/aioXXXXXX"));
  mempcpy (mempcpy (name, test_dir, name_len),
	   "/aioXXXXXX", sizeof ("/aioXXXXXX"));

  /* Open our test file.   */
  fd = mkstemp (name);
  if (fd == -1)
    error (EXIT_FAILURE, errno, "cannot open test file `%s'", name);
  add_temp_file (name);
}


static int
test_file (const void *buf, size_t size, int fd, const char *msg)
{
  struct stat st;
  char tmp[size];

  errno = 0;
  if (fstat (fd, &st) < 0)
    {
      error (0, errno, "%s: failed stat", msg);
      return 1;
    }

  if (st.st_size != (off_t) size)
    {
      error (0, errno, "%s: wrong size: %lu, should be %lu",
	     msg, (unsigned long int) st.st_size, (unsigned long int) size);
      return 1;
    }

  if (pread (fd, tmp, size, 0) != (ssize_t) size)
    {
      error (0, errno, "%s: failed pread", msg);
      return 1;
    }

  if (memcmp (buf, tmp, size) != 0)
    {
      error (0, errno, "%s: failed comparison", msg);
      return 1;
    }

  printf ("%s test ok\n", msg);

  return 0;
}


static int
do_wait (struct aiocb **cbp, size_t nent, int allowed_err)
{
  int go_on;
  size_t cnt;
  int result = 0;

  do
    {
      aio_suspend ((const struct aiocb *const *) cbp, nent, NULL);
      go_on = 0;
      for (cnt = 0; cnt < nent; ++cnt)
	if (cbp[cnt] != NULL)
	  {
	    if (aio_error (cbp[cnt]) == EINPROGRESS)
	      go_on = 1;
	    else
	      {
		if (aio_return (cbp[cnt]) == -1
		    && (allowed_err == 0
			|| aio_error (cbp[cnt]) != allowed_err))
		  {
		    error (0, aio_error (cbp[cnt]), "Operation failed\n");
		    result = 1;
		  }
		cbp[cnt] = NULL;
	      }
	  }
    }
  while (go_on);

  return result;
}


int
do_test (int argc, char *argv[])
{
  struct aiocb cbs[10];
  struct aiocb cbs_fsync;
  struct aiocb *cbp[10];
  struct aiocb *cbp_fsync[1];
  char buf[1000];
  size_t cnt;
  int result = 0;

  /* Preparation.  */
  for (cnt = 0; cnt < 10; ++cnt)
    {
      cbs[cnt].aio_fildes = fd;
      cbs[cnt].aio_reqprio = 0;
      cbs[cnt].aio_buf = memset (&buf[cnt * 100], '0' + cnt, 100);
      cbs[cnt].aio_nbytes = 100;
      cbs[cnt].aio_offset = cnt * 100;
      cbs[cnt].aio_sigevent.sigev_notify = SIGEV_NONE;

      cbp[cnt] = &cbs[cnt];
    }

  /* First a simple test.  */
  for (cnt = 10; cnt > 0; )
    if (aio_write (cbp[--cnt]) < 0 && errno == ENOSYS)
      {
	error (0, 0, "no aio support in this configuration");
	return 0;
      }
  /* Wait 'til the results are there.  */
  result |= do_wait (cbp, 10, 0);
  /* Test this.  */
  result |= test_file (buf, sizeof (buf), fd, "aio_write");

  /* Read now as we've written it.  */
  memset (buf, '\0', sizeof (buf));
  /* Issue the commands.  */
  for (cnt = 10; cnt > 0; )
    {
      --cnt;
      cbp[cnt] = &cbs[cnt];
      aio_read (cbp[cnt]);
    }
  /* Wait 'til the results are there.  */
  result |= do_wait (cbp, 10, 0);
  /* Test this.  */
  for (cnt = 0; cnt < 1000; ++cnt)
    if (buf[cnt] != '0' + (cnt / 100))
      {
	result = 1;
	error (0, 0, "comparison failed for aio_read test");
	break;
      }

  if (cnt == 1000)
    puts ("aio_read test ok");

  /* Remove the test file contents.  */
  if (ftruncate (fd, 0) < 0)
    {
      error (0, errno, "ftruncate failed\n");
      result = 1;
    }

  /* Test lio_listio.  */
  for (cnt = 0; cnt < 10; ++cnt)
    {
      cbs[cnt].aio_lio_opcode = LIO_WRITE;
      cbp[cnt] = &cbs[cnt];
    }
  /* Issue the command.  */
  lio_listio (LIO_WAIT, cbp, 10, NULL);
  /* ...and immediately test it since we started it in wait mode.  */
  result |= test_file (buf, sizeof (buf), fd, "lio_listio (write)");

  /* Test aio_fsync.  */
  cbs_fsync.aio_fildes = fd;
  cbs_fsync.aio_sigevent.sigev_notify = SIGEV_NONE;
  cbp_fsync[0] = &cbs_fsync;

  /* Remove the test file contents first.  */
  if (ftruncate (fd, 0) < 0)
    {
      error (0, errno, "ftruncate failed\n");
      result = 1;
    }

  /* Write again.  */
  for (cnt = 10; cnt > 0; )
    aio_write (cbp[--cnt]);

  if (aio_fsync (O_SYNC, &cbs_fsync) < 0)
    {
      error (0, errno, "aio_fsync failed\n");
      result = 1;
    }
  result |= do_wait (cbp_fsync, 1, 0);

  /* ...and test since all data should be on disk now.  */
  result |= test_file (buf, sizeof (buf), fd, "aio_fsync (aio_write)");

  /* Test aio_cancel.  */
  /* Remove the test file contents first.  */
  if (ftruncate (fd, 0) < 0)
    {
      error (0, errno, "ftruncate failed\n");
      result = 1;
    }

  /* Write again.  */
  for (cnt = 10; cnt > 0; )
    aio_write (cbp[--cnt]);

  /* Cancel all requests.  */
  if (aio_cancel (fd, NULL) == -1)
    printf ("aio_cancel (fd, NULL) cannot cancel anything\n");

  result |= do_wait (cbp, 10, ECANCELED);

  /* Another test for aio_cancel.  */
  /* Remove the test file contents first.  */
  if (ftruncate (fd, 0) < 0)
    {
      error (0, errno, "ftruncate failed\n");
      result = 1;
    }

  /* Write again.  */
  for (cnt = 10; cnt > 0; )
    {
      --cnt;
      cbp[cnt] = &cbs[cnt];
      aio_write (cbp[cnt]);
    }
  puts ("finished3");

  /* Cancel all requests.  */
  for (cnt = 10; cnt > 0; )
    if (aio_cancel (fd, cbp[--cnt]) == -1)
      /* This is not an error.  The request can simply be finished.  */
      printf ("aio_cancel (fd, cbp[%Zd]) cannot be canceled\n", cnt);
  puts ("finished2");

  result |= do_wait (cbp, 10, ECANCELED);

  puts ("finished");

  return result;
}
