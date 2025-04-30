/* Create subprocess.
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

#include <stdio.h>
#include <signal.h>
#include <time.h>
#include <sys/wait.h>
#include <stdbool.h>
#include <support/xspawn.h>
#include <support/check.h>
#include <support/xunistd.h>
#include <support/subprocess.h>

static struct support_subprocess
support_subprocess_init (void)
{
  struct support_subprocess result;

  xpipe (result.stdout_pipe);
  TEST_VERIFY (result.stdout_pipe[0] > STDERR_FILENO);
  TEST_VERIFY (result.stdout_pipe[1] > STDERR_FILENO);

  xpipe (result.stderr_pipe);
  TEST_VERIFY (result.stderr_pipe[0] > STDERR_FILENO);
  TEST_VERIFY (result.stderr_pipe[1] > STDERR_FILENO);

  TEST_VERIFY (fflush (stdout) == 0);
  TEST_VERIFY (fflush (stderr) == 0);

  return result;
}

struct support_subprocess
support_subprocess (void (*callback) (void *), void *closure)
{
  struct support_subprocess result = support_subprocess_init ();

  result.pid = xfork ();
  if (result.pid == 0)
    {
      xclose (result.stdout_pipe[0]);
      xclose (result.stderr_pipe[0]);
      xdup2 (result.stdout_pipe[1], STDOUT_FILENO);
      xdup2 (result.stderr_pipe[1], STDERR_FILENO);
      xclose (result.stdout_pipe[1]);
      xclose (result.stderr_pipe[1]);
      callback (closure);
      _exit (0);
    }
  xclose (result.stdout_pipe[1]);
  xclose (result.stderr_pipe[1]);

  return result;
}

struct support_subprocess
support_subprogram (const char *file, char *const argv[])
{
  struct support_subprocess result = support_subprocess_init ();

  posix_spawn_file_actions_t fa;
  /* posix_spawn_file_actions_init does not fail.  */
  posix_spawn_file_actions_init (&fa);

  xposix_spawn_file_actions_addclose (&fa, result.stdout_pipe[0]);
  xposix_spawn_file_actions_addclose (&fa, result.stderr_pipe[0]);
  xposix_spawn_file_actions_adddup2 (&fa, result.stdout_pipe[1], STDOUT_FILENO);
  xposix_spawn_file_actions_adddup2 (&fa, result.stderr_pipe[1], STDERR_FILENO);
  xposix_spawn_file_actions_addclose (&fa, result.stdout_pipe[1]);
  xposix_spawn_file_actions_addclose (&fa, result.stderr_pipe[1]);

  result.pid = xposix_spawn (file, &fa, NULL, argv, environ);

  xclose (result.stdout_pipe[1]);
  xclose (result.stderr_pipe[1]);

  return result;
}

int
support_subprogram_wait (const char *file, char *const argv[])
{
  posix_spawn_file_actions_t fa;

  posix_spawn_file_actions_init (&fa);
  struct support_subprocess res = support_subprocess_init ();

  res.pid = xposix_spawn (file, &fa, NULL, argv, environ);

  return support_process_wait (&res);
}

int
support_process_wait (struct support_subprocess *proc)
{
  xclose (proc->stdout_pipe[0]);
  xclose (proc->stderr_pipe[0]);

  int status;
  xwaitpid (proc->pid, &status, 0);
  return status;
}


static bool
support_process_kill (int pid, int signo, int *status)
{
  /* Kill the whole process group.  */
  kill (-pid, signo);
  /* In case setpgid failed in the child, kill it individually too.  */
  kill (pid, signo);

  /* Wait for it to terminate.  */
  pid_t killed;
  for (int i = 0; i < 5; ++i)
    {
      int status;
      killed = xwaitpid (pid, &status, WNOHANG|WUNTRACED);
      if (killed != 0)
        break;

      /* Delay, give the system time to process the kill.  If the
         nanosleep() call return prematurely, all the better.  We
         won't restart it since this probably means the child process
         finally died.  */
      nanosleep (&((struct timespec) { 0, 100000000 }), NULL);
    }
  if (killed != 0 && killed != pid)
    return false;

  return true;
}

int
support_process_terminate (struct support_subprocess *proc)
{
  xclose (proc->stdout_pipe[0]);
  xclose (proc->stderr_pipe[0]);

  int status;
  pid_t killed = xwaitpid (proc->pid, &status, WNOHANG|WUNTRACED);
  if (killed != 0 && killed == proc->pid)
    return status;

  /* Subprocess is still running, terminate it.  */
  if (!support_process_kill (proc->pid, SIGTERM, &status) )
    support_process_kill (proc->pid, SIGKILL, &status);

  return status;
}
