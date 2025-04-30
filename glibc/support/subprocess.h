/* Create a subprocess.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_SUBPROCESS_H
#define SUPPORT_SUBPROCESS_H

#include <sys/types.h>

struct support_subprocess
{
  int stdout_pipe[2];
  int stderr_pipe[2];
  pid_t pid;
};

/* Invoke CALLBACK (CLOSURE) in a subprocess created with fork and return
   its PID, a pipe redirected to STDOUT, and a pipe redirected to STDERR.  */
struct support_subprocess support_subprocess
  (void (*callback) (void *), void *closure);

/* Issue FILE with ARGV arguments by using posix_spawn and return is PID, a
   pipe redirected to STDOUT, and a pipe redirected to STDERR.  */
struct support_subprocess support_subprogram
  (const char *file, char *const argv[]);

/* Invoke program FILE with ARGV arguments by using posix_spawn and wait for it
   to complete.  Return program exit status.  */
int support_subprogram_wait
  (const char *file, char *const argv[]);

/* Wait for the subprocess indicated by PROC::PID.  Return the status
   indicate by waitpid call.  */
int support_process_wait (struct support_subprocess *proc);

/* Terminate the subprocess indicated by PROC::PID, first with a SIGTERM and
   then with a SIGKILL.  Return the status as for waitpid call.  */
int support_process_terminate (struct support_subprocess *proc);

#endif
