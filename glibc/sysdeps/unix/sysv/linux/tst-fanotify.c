/* Basic fanotify test.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <config.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/fanotify.h>

static int
do_test (void)
{
  int fd, ret;

  fd = fanotify_init (0, 0);
  if (fd < 0)
    {
      switch (errno)
	{
	case ENOSYS:
	  puts ("SKIP: missing support for fanotify (check CONFIG_FANOTIFY=y)");
	  return 0;
	case EPERM:
	  puts ("SKIP: missing proper permissions for runtime test");
	  return 0;
	}

      perror ("fanotify_init (0, 0) failed");
      return 1;
    }

  ret = fanotify_mark (fd, FAN_MARK_ADD | FAN_MARK_MOUNT, FAN_ACCESS
		       | FAN_MODIFY | FAN_OPEN | FAN_CLOSE | FAN_ONDIR
		       | FAN_EVENT_ON_CHILD, AT_FDCWD, ".");
  if (ret)
    {
      perror ("fanotify_mark (...) failed");
      return 1;
    }

  puts ("All OK");
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
