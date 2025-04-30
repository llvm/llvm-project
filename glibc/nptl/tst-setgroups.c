/* Test setgroups as root and in the presence of threads (Bug 26248)
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <limits.h>
#include <grp.h>
#include <errno.h>
#include <error.h>
#include <support/xthread.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xunistd.h>

/* The purpose of this test is to test the setgroups API as root and in
   the presence of threads.  Once we create a thread the setgroups
   implementation must ensure that all threads are set to the same
   group and this operation should not fail. Lastly we test setgroups
   with a zero sized group and a bad address and verify we get EPERM.  */

static void *
start_routine (void *args)
{
  return NULL;
}

static int
do_test (void)
{
  int size;
  /* NB: Stack address can be at 0xfffXXXXX on 32-bit OSes.  */
  gid_t list[NGROUPS_MAX];
  int status = EXIT_SUCCESS;

  pthread_t thread = xpthread_create (NULL, start_routine, NULL);

  size = getgroups (sizeof (list) / sizeof (list[0]), list);
  if (size < 0)
    {
      status = EXIT_FAILURE;
      error (0, errno, "getgroups failed");
    }
  if (setgroups (size, list) < 0)
    {
      if (errno == EPERM)
	status = EXIT_UNSUPPORTED;
      else
	{
	  status = EXIT_FAILURE;
	  error (0, errno, "setgroups failed");
	}
    }

  if (status == EXIT_SUCCESS && setgroups (0, list) < 0)
    {
      status = EXIT_FAILURE;
      error (0, errno, "setgroups failed");
    }

  xpthread_join (thread);

  return status;
}

#include <support/test-driver.c>
