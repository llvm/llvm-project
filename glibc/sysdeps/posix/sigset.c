/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#define __need_NULL
#include <stddef.h>
#include <signal.h>
#include <string.h>	/* For the real memset prototype.  */
#include <sigsetops.h>

/* Set the disposition for SIG.  */
__sighandler_t
sigset (int sig, __sighandler_t disp)
{
  struct sigaction act;
  struct sigaction oact;
  sigset_t set;
  sigset_t oset;

  __sigemptyset (&set);
  if (sigaddset (&set, sig) < 0)
    return SIG_ERR;

  if (disp == SIG_HOLD)
    {
      /* Add the signal to the current signal mask.  */
      if (__sigprocmask (SIG_BLOCK, &set, &oset) < 0)
	return SIG_ERR;

      /* If the signal was already blocked signal this to the caller.  */
      if (__sigismember (&oset, sig))
	return SIG_HOLD;

      /* We need to determine whether a specific handler is installed.  */
      if (__sigaction (sig, NULL, &oact) < 0)
	return SIG_ERR;

      return oact.sa_handler;
    }
  else
    {
      act.sa_handler = disp;
      __sigemptyset (&act.sa_mask);
      act.sa_flags = 0;
      if (__sigaction (sig, &act, &oact) < 0)
	return SIG_ERR;

      /* Remove the signal from the current signal mask.  */
      if (__sigprocmask (SIG_UNBLOCK, &set, &oset) < 0)
	return SIG_ERR;

      /* If the signal was already blocked return SIG_HOLD.  */
      return __sigismember (&oset, sig) ? SIG_HOLD : oact.sa_handler;
    }
}
