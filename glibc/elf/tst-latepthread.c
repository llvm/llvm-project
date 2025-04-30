/* Test that loading libpthread does not break ld.so exceptions (bug 16628).
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

#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int
do_test (void)
{
  void *handle = dlopen ("tst-latepthreadmod.so", RTLD_LOCAL | RTLD_LAZY);
  if (handle == NULL)
    {
      printf ("error: dlopen failed: %s\n", dlerror ());
      return 1;
    }
  void *ptr = dlsym (handle, "trigger_dynlink_failure");
  if (ptr == NULL)
    {
      printf ("error: dlsym failed: %s\n", dlerror ());
      return 1;
    }
  int (*func) (void) = ptr;

  /* Run the actual test in a subprocess, to capture the error.  */
  int fds[2];
  if (pipe (fds) < 0)
    {
      printf ("error: pipe: %m\n");
      return 1;
    }
  pid_t pid = fork ();
  if (pid < 0)
    {
      printf ("error: fork: %m\n");
      return 1;
    }
  else if (pid == 0)
    {
      if (dup2 (fds[1], STDERR_FILENO) < 0)
        _exit (2);
      /* Trigger an abort.  */
      func ();
      _exit (3);
    }
  /* NB: This assumes that the abort message is so short that the pipe
     does not block.  */
  int status;
  if (waitpid (pid, &status, 0) < 0)
    {
      printf ("error: waitpid: %m\n");
      return 1;
    }

  /* Check the printed error message.  */
  if (close (fds[1]) < 0)
   {
     printf ("error: close: %m\n");
     return 1;
   }
  char buf[512];
  /* Leave room for the NUL terminator.  */
  ssize_t ret = read (fds[0], buf, sizeof (buf) - 1);
  if (ret < 0)
    {
      printf ("error: read: %m\n");
      return 1;
    }
  if (ret > 0 && buf[ret - 1] == '\n')
    --ret;
  buf[ret] = '\0';
  printf ("info: exit status: %d, message: %s\n", status, buf);
  if (strstr (buf, "undefined symbol: this_function_is_not_defined") == NULL)
    {
      printf ("error: message does not contain expected string\n");
      return 1;
    }
  if (!WIFEXITED (status) || WEXITSTATUS (status) != 127)
    {
      printf ("error: unexpected process exit status\n");
      return 1;
    }
  return 0;
}

#include <support/test-driver.c>
