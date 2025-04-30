/* Test posix_spawn setsid attribute.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <fcntl.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

#include <support/check.h>

static void
do_test_setsid (bool test_setsid)
{
  pid_t sid, child_sid;
  int res;

  /* Current session ID.  */
  sid = getsid(0);
  if (sid == (pid_t) -1)
    FAIL_EXIT1 ("getsid (0): %m");

  posix_spawnattr_t attrp;
  /* posix_spawnattr_init should not fail (it basically memset the
     attribute).  */
  posix_spawnattr_init (&attrp);
  if (test_setsid)
    {
      res = posix_spawnattr_setflags (&attrp, POSIX_SPAWN_SETSID);
      if (res != 0)
	{
	  errno = res;
	  FAIL_EXIT1 ("posix_spawnattr_setflags: %m");
	}
    }

  /* Program to run.  */
  char *args[2] = { (char *) "true", NULL };
  pid_t child;

  res = posix_spawnp (&child, "true", NULL, &attrp, args, environ);
  /* posix_spawnattr_destroy is noop.  */
  posix_spawnattr_destroy (&attrp);

  if (res != 0)
    {
      errno = res;
      FAIL_EXIT1 ("posix_spawnp: %m");
    }

  /* Child should have a different session ID than parent.  */
  child_sid = getsid (child);

  if (child_sid == (pid_t) -1)
    FAIL_EXIT1 ("getsid (%i): %m", child);

  if (test_setsid)
    {
      if (child_sid == sid)
	FAIL_EXIT1 ("child session ID matched parent one");
    }
  else
    {
      if (child_sid != sid)
	FAIL_EXIT1 ("child session ID did not match parent one");
    }
}

static int
do_test (void)
{
  do_test_setsid (false);
  do_test_setsid (true);

  return 0;
}

#include <support/test-driver.c>
