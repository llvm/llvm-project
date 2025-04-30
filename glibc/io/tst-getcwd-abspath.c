/* BZ #22679 getcwd(3) should not succeed without returning an absolute path.

   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xunistd.h>
#include <unistd.h>

static char *chroot_dir;

/* The actual test.  Run it in a subprocess, so that the test harness
   can remove the temporary directory in --direct mode.  */
static void
getcwd_callback (void *closure)
{
  xchroot (chroot_dir);

  errno = 0;
  char *cwd = getcwd (NULL, 0);
  TEST_COMPARE (errno, ENOENT);
  TEST_VERIFY (cwd == NULL);

  errno = 0;
  cwd = realpath (".", NULL);
  TEST_COMPARE (errno, ENOENT);
  TEST_VERIFY (cwd == NULL);

  _exit (0);
}

static int
do_test (void)
{
  support_become_root ();
  if (!support_can_chroot ())
    return EXIT_UNSUPPORTED;

  chroot_dir = support_create_temp_directory ("tst-getcwd-abspath-");
  support_isolate_in_subprocess (getcwd_callback, NULL);

  return 0;
}

#include <support/test-driver.c>
