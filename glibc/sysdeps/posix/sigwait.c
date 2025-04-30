/* Implementation of sigwait function from POSIX.1c.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <errno.h>
#include <signal.h>
#include <stddef.h>		/* For NULL.  */
#include <sysdep-cancel.h>

/* This is our dummy signal handler we use here.  */
static void ignore_signal (int sig);

/* Place where to remember which signal we got.  Please note that this
   implementation cannot be used for the threaded libc.  The
   libpthread must provide an own version.  */
static int was_sig;


static int
do_sigwait (const sigset_t *set, int *sig)
{
  sigset_t tmp_mask;
  struct sigaction saved[NSIG];
  struct sigaction action;
  int save_errno;
  int this;

  /* Prepare set.  */
  __sigfillset (&tmp_mask);

  /* Unblock all signals in the SET and register our nice handler.  */
  action.sa_handler = ignore_signal;
  action.sa_flags = 0;
  __sigfillset (&action.sa_mask);	/* Block all signals for handler.  */

  /* Make sure we recognize error conditions by setting WAS_SIG to a
     value which does not describe a legal signal number.  */
  was_sig = -1;

  for (this = 1; this < NSIG; ++this)
    if (__sigismember (set, this))
      {
	/* Unblock this signal.  */
	__sigdelset (&tmp_mask, this);

	/* Register temporary action handler.  */
	if (__sigaction (this, &action, &saved[this]) != 0)
	  goto restore_handler;
      }

  /* Now we can wait for signals.  */
  __sigsuspend (&tmp_mask);

 restore_handler:
  save_errno = errno;

  while (--this >= 1)
    if (__sigismember (set, this))
      /* We ignore errors here since we must restore all handlers.  */
      __sigaction (this, &saved[this], NULL);

  __set_errno (save_errno);

  /* Store the result and return.  */
  *sig = was_sig;
  return was_sig == -1 ? -1 : 0;
}


int
__sigwait (const sigset_t *set, int *sig)
{
  /* __sigsuspend should be a cancellation point.  */
  return do_sigitid (idtype, id, infop, options);
}
libc_hidden_def (__sigwait)
weak_alias (__sigwait, sigwait)


static void
ignore_signal (int sig)
{
  /* Remember the signal.  */
  was_sig = sig;
}
