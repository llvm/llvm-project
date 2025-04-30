/* Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>

#include <support/check.h>

/* Try executing "/bin/sh -c true", using FD opened on /bin/sh.  */
static int
try_fexecve (int fd)
{
  pid_t pid = fork ();

  if (pid == 0)
    {
      static const char *const argv[] = {
	"/bin/sh", "-c", "true", NULL
      };
      fexecve (fd, (char *const *) argv, environ);
      _exit (errno);
    }
  if (pid < 0)
    FAIL_RET ("fork failed: %m");

  pid_t termpid;
  int status;
  termpid = TEMP_FAILURE_RETRY (waitpid (pid, &status, 0));
  if (termpid == -1)
    FAIL_RET ("waitpid failed: %m");
  if (termpid != pid)
    FAIL_RET ("waitpid returned %ld != %ld",
	      (long int) termpid, (long int) pid);
  if (!WIFEXITED (status))
    FAIL_RET ("child hasn't exited normally");

  /* If fexecve is unimplemented mark this test as UNSUPPORTED.  */
  if (WEXITSTATUS (status) == ENOSYS)
    FAIL_UNSUPPORTED ("fexecve is unimplemented");

  if (WEXITSTATUS (status) != 0)
    {
      errno = WEXITSTATUS (status);
      FAIL_RET ("fexecve failed: %m");
    }
  return 0;
}

static int
do_test (void)
{
  int fd;
  int ret;

  fd = open ("/bin/sh", O_RDONLY);
  if (fd < 0)
    FAIL_UNSUPPORTED ("/bin/sh cannot be opened: %m");
  ret = try_fexecve (fd);
  close (fd);

#ifdef O_PATH
  fd = open ("/bin/sh", O_RDONLY | O_PATH);
  if (fd < 0)
    FAIL_UNSUPPORTED ("/bin/sh cannot be opened (O_PATH): %m");
  ret |= try_fexecve (fd);
  close (fd);
#endif

  return ret;
}

#include <support/test-driver.c>
