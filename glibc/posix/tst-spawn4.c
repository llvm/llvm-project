/* Check if posix_spawn does handle correctly ENOEXEC files.
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

#include <spawn.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>

#include <support/xunistd.h>
#include <support/check.h>
#include <support/temp_file.h>

static int
do_test (void)
{
  char *scriptname;
  int fd = create_temp_file ("tst-spawn4.", &scriptname);
  TEST_VERIFY_EXIT (fd >= 0);

  const char script[] = "echo it should not happen";
  xwrite (fd, script, sizeof (script) - 1);
  xclose (fd);

  TEST_VERIFY_EXIT (chmod (scriptname, 0x775) == 0);

  pid_t pid;
  int status;

  /* Check if scripts without shebang are correctly not executed.  */
  status = posix_spawn (&pid, scriptname, NULL, NULL, (char *[]) { 0 },
                        (char *[]) { 0 });
  TEST_VERIFY_EXIT (status == ENOEXEC);

  status = posix_spawnp (&pid, scriptname, NULL, NULL, (char *[]) { 0 },
                         (char *[]) { 0 });
  TEST_VERIFY_EXIT (status == ENOEXEC);

  return 0;
}

#include <support/test-driver.c>
