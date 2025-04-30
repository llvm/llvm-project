/* Bug 18125: Verify setcontext calls exit() and not _exit().
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

/* Please note that depending on the outcome of Bug 18135 this test
   may become invalid, and instead of testing for calling exit it
   should be reworked to test for the last context calling
   pthread_exit().  */

static ucontext_t ctx;
static char *filename;

/* It is intended that this function does nothing.  */
static void
cf (void)
{
  printf ("called context function\n");
}

static void
exit_called (void)
{
  int fd;
  ssize_t res;
  const char buf[] = "Called exit function\n";

  fd = open (filename, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
  if (fd == -1)
    {
      printf ("FAIL: Unable to create test file %s\n", filename);
      exit (1);
    }
  res = write (fd, buf, sizeof (buf));
  if (res != sizeof (buf))
    {
      printf ("FAIL: Expected to write test file in one write call.\n");
      exit (1);
    }
  res = close (fd);
  if (res == -1)
    {
      printf ("FAIL: Failed to close test file.\n");
      exit (1);
    }
  printf ("PASS: %s", buf);
}

/* The test expects a filename given by the wrapper calling script.
   The test then registers an atexit handler that will create the
   file to indicate that the atexit handler ran. Then the test
   creates a context, modifies it with makecontext, and sets it.
   The context has only a single context which then must exit.
   If it incorrectly exits via _exit then the atexit handler is
   not run, the file is not created, and the wrapper detects this
   and fails the test.  This test cannot be done using an _exit
   interposer since setcontext avoids the PLT and calls _exit
   directly.  */
static int
do_test (int argc, char **argv)
{
  int ret;
  char st1[32768];
  ucontext_t tempctx = ctx;

  if (argc < 2)
    {
      printf ("FAIL: Test missing filename argument.\n");
      exit (1);
    }

  filename = argv[1];

  atexit (exit_called);

  puts ("making contexts");
  if (getcontext (&ctx) != 0)
    {
      if (errno == ENOSYS)
	{
	  /* Exit with 77 to mark the test as UNSUPPORTED.  */
	  printf ("UNSUPPORTED: getcontext not implemented.\n");
	  exit (77);
	}

      printf ("FAIL: getcontext failed.\n");
      exit (1);
    }

  ctx.uc_stack.ss_sp = st1;
  ctx.uc_stack.ss_size = sizeof (st1);
  ctx.uc_link = 0;
  makecontext (&ctx, cf, 0);

  /* Without this check, a stub makecontext can make us spin forever.  */
  if (memcmp (&tempctx, &ctx, sizeof ctx) == 0)
    {
      puts ("UNSUPPORTED: makecontext was a no-op, presuming not implemented");
      exit (77);
    }

  ret = setcontext (&ctx);
  if (ret != 0)
    {
      printf ("FAIL: setcontext returned with %d and errno of %d.\n", ret, errno);
      exit (1);
    }

  printf ("FAIL: Impossibly returned to main.\n");
  exit (1);
}

#include "../test-skeleton.c"
