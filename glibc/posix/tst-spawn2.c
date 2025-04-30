/* Further tests for spawn in case of invalid binary paths.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
#include <error.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdio.h>

#include <support/check.h>

int
do_test (void)
{
  /* Check if posix_spawn correctly returns an error and an invalid pid
     by trying to spawn an invalid binary.  */

  const char *program = "/path/to/invalid/binary";
  char * const args[] = { 0 };
  pid_t pid = -1;

  int ret = posix_spawn (&pid, program, 0, 0, args, environ);
  if (ret != ENOENT)
    {
      errno = ret;
      FAIL_EXIT1 ("posix_spawn: %m");
    }

  /* POSIX states the value returned on pid variable in case of an error
     is not specified.  GLIBC will update the value iff the child
     execution is successful.  */
  if (pid != -1)
    FAIL_EXIT1 ("posix_spawn returned pid != -1 (%i)", (int) pid);

  /* Check if no child is actually created.  */
  ret = waitpid (-1, NULL, 0);
  if (ret != -1 || errno != ECHILD)
    FAIL_EXIT1 ("waitpid: %m)");

  /* Same as before, but with posix_spawnp.  */
  char *args2[] = { (char*) program, 0 };

  ret = posix_spawnp (&pid, args2[0], 0, 0, args2, environ);
  if (ret != ENOENT)
    {
      errno = ret;
      FAIL_EXIT1 ("posix_spawnp: %m");
    }

  if (pid != -1)
    FAIL_EXIT1 ("posix_spawnp returned pid != -1 (%i)", (int) pid);

  ret = waitpid (-1, NULL, 0);
  if (ret != -1 || errno != ECHILD)
    FAIL_EXIT1 ("waitpid: %m)");

  return 0;
}

#include <support/test-driver.c>
