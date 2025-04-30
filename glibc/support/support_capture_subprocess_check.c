/* Verify capture output from a subprocess.
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

#include <stdbool.h>
#include <stdio.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <sys/wait.h>

static void
print_context (const char *context, bool *failed)
{
  if (*failed)
    /* Do not duplicate message.  */
    return;
  support_record_failure ();
  printf ("error: subprocess failed: %s\n", context);
}

static void
print_actual_status (struct support_capture_subprocess *proc)
{
  if (WIFEXITED (proc->status))
    printf ("error:   actual exit status: %d [0x%x]\n",
            WEXITSTATUS (proc->status), proc->status);
  else if (WIFSIGNALED (proc->status))
    printf ("error:   actual termination signal: %d [0x%x]\n",
            WTERMSIG (proc->status), proc->status);
  else
    printf ("error:   actual undecoded exit status: [0x%x]\n", proc->status);
}

void
support_capture_subprocess_check (struct support_capture_subprocess *proc,
                                  const char *context, int status_or_signal,
                                  int allowed)
{
  TEST_VERIFY ((allowed & sc_allow_none)
               || (allowed & sc_allow_stdout)
               || (allowed & sc_allow_stderr));
  TEST_VERIFY (!((allowed & sc_allow_none)
                 && ((allowed & sc_allow_stdout)
                     || (allowed & sc_allow_stderr))));

  bool failed = false;
  if (status_or_signal >= 0)
    {
      /* Expect regular termination.  */
      if (!(WIFEXITED (proc->status)
            && WEXITSTATUS (proc->status) == status_or_signal))
        {
          print_context (context, &failed);
          printf ("error:   expected exit status: %d\n", status_or_signal);
          print_actual_status (proc);
        }
    }
  else
    {
      /* status_or_signal < 0.  Expect termination by signal.  */
      if (!(WIFSIGNALED (proc->status)
            && WTERMSIG (proc->status) == -status_or_signal))
        {
          print_context (context, &failed);
          printf ("error:   expected termination signal: %d\n",
                  -status_or_signal);
          print_actual_status (proc);
        }
    }
  if (!(allowed & sc_allow_stdout) && proc->out.length != 0)
    {
      print_context (context, &failed);
      printf ("error:   unexpected output from subprocess\n");
      fwrite (proc->out.buffer, proc->out.length, 1, stdout);
      puts ("\n");
    }
  if (!(allowed & sc_allow_stderr) && proc->err.length != 0)
    {
      print_context (context, &failed);
      printf ("error:   unexpected error output from subprocess\n");
      fwrite (proc->err.buffer, proc->err.length, 1, stdout);
      puts ("\n");
    }
}
