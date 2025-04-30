/* Invoke the system diff tool to compare two strings.
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

#include <support/run_diff.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xunistd.h>
#include <sys/wait.h>

static char *
write_to_temp_file (const char *prefix, const char *str)
{
  char *template = xasprintf ("run_diff-%s", prefix);
  char *name = NULL;
  int fd = create_temp_file (template, &name);
  TEST_VERIFY_EXIT (fd >= 0);
  free (template);
  xwrite (fd, str, strlen (str));
  xclose (fd);
  return name;
}

void
support_run_diff (const char *left_label, const char *left,
                  const char *right_label, const char *right)
{
  /* Ensure that the diff command output is ordered properly with
     standard output.  */
  TEST_VERIFY_EXIT (fflush (stdout) == 0);

  char *left_path = write_to_temp_file ("left-diff", left);
  char *right_path = write_to_temp_file ("right-diff", right);

  pid_t pid = xfork ();
  if (pid == 0)
    {
      execlp ("diff", "diff", "-u",
              "--label", left_label, "--label", right_label,
              "--", left_path, right_path,
              NULL);
      _exit (17);
    }
  else
    {
      int status;
      xwaitpid (pid, &status, 0);
      if (!WIFEXITED (status) || WEXITSTATUS (status) != 1)
        printf ("warning: could not run diff, exit status: %d\n"
                "*** %s ***\n%s\n"
                "*** %s ***\n%s\n",
                status, left_label, left, right_label, right);
    }

  free (right_path);
  free (left_path);
}
