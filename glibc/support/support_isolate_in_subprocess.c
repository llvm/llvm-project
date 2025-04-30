/* Run a function in a subprocess.
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

#include <support/check.h>
#include <support/xunistd.h>

void
support_isolate_in_subprocess (void (*callback) (void *), void *closure)
{
  pid_t pid = xfork ();
  if (pid == 0)
    {
      /* Child process.  */
      callback (closure);
      _exit (0);
    }

  /* Parent process.  */
  int status;
  xwaitpid (pid, &status, 0);
  if (status != 0)
    FAIL_EXIT1 ("child process exited with status %d", status);
}
