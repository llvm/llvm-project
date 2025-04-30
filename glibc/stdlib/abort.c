/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <libc-lock.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sigsetops.h>

/* Try to get a machine dependent instruction which will make the
   program crash.  This is used in case everything else fails.  */
#include <abort-instr.h>
#ifndef ABORT_INSTRUCTION
/* No such instruction is available.  */
# define ABORT_INSTRUCTION
#endif

/* Exported variable to locate abort message in core files etc.  */
struct abort_msg_s *__abort_msg;
libc_hidden_def (__abort_msg)

/* We must avoid to run in circles.  Therefore we remember how far we
   already got.  */
static int stage;

/* We should be prepared for multiple threads trying to run abort.  */
__libc_lock_define_initialized_recursive (static, lock);


/* Cause an abnormal program termination with core-dump.  */
void
abort (void)
{
  struct sigaction act;
  sigset_t sigs;

  /* First acquire the lock.  */
  __libc_lock_lock_recursive (lock);

  /* Now it's for sure we are alone.  But recursive calls are possible.  */

  /* Unblock SIGABRT.  */
  if (stage == 0)
    {
      ++stage;
      __sigemptyset (&sigs);
      __sigaddset (&sigs, SIGABRT);
      __sigprocmask (SIG_UNBLOCK, &sigs, 0);
    }

  /* Send signal which possibly calls a user handler.  */
  if (stage == 1)
    {
      /* This stage is special: we must allow repeated calls of
	 `abort' when a user defined handler for SIGABRT is installed.
	 This is risky since the `raise' implementation might also
	 fail but I don't see another possibility.  */
      int save_stage = stage;

      stage = 0;
      __libc_lock_unlock_recursive (lock);

      raise (SIGABRT);

      __libc_lock_lock_recursive (lock);
      stage = save_stage + 1;
    }

  /* There was a handler installed.  Now remove it.  */
  if (stage == 2)
    {
      ++stage;
      memset (&act, '\0', sizeof (struct sigaction));
      act.sa_handler = SIG_DFL;
      __sigfillset (&act.sa_mask);
      act.sa_flags = 0;
      __sigaction (SIGABRT, &act, NULL);
    }

  /* Try again.  */
  if (stage == 3)
    {
      ++stage;
      raise (SIGABRT);
    }

  /* Now try to abort using the system specific command.  */
  if (stage == 4)
    {
      ++stage;
      ABORT_INSTRUCTION;
    }

  /* If we can't signal ourselves and the abort instruction failed, exit.  */
  if (stage == 5)
    {
      ++stage;
      _exit (127);
    }

  /* If even this fails try to use the provided instruction to crash
     or otherwise make sure we never return.  */
  while (1)
    /* Try for ever and ever.  */
    ABORT_INSTRUCTION;
}
libc_hidden_def (abort)
