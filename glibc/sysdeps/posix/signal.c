/* BSD-like signal function.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <signal.h>
#include <sigsetops.h>
#include <internal-signals.h>

sigset_t _sigintr attribute_hidden;		/* Set by siginterrupt.  */

/* Set the handler for the signal SIG to HANDLER,
   returning the old handler, or SIG_ERR on error.  */
__sighandler_t
__bsd_signal (int sig, __sighandler_t handler)
{
  struct sigaction act, oact;

  /* Check signal extents to protect __sigismember.  */
  if (handler == SIG_ERR || sig < 1 || sig >= NSIG
      || __is_internal_signal (sig))
    {
      __set_errno (EINVAL);
      return SIG_ERR;
    }

  act.sa_handler = handler;
  __sigemptyset (&act.sa_mask);
  __sigaddset (&act.sa_mask, sig);
  act.sa_flags = __sigismember (&_sigintr, sig) ? 0 : SA_RESTART;
  if (__sigaction (sig, &act, &oact) < 0)
    return SIG_ERR;

  return oact.sa_handler;
}
weak_alias (__bsd_signal, bsd_signal)
weak_alias (__bsd_signal, signal)
weak_alias (__bsd_signal, ssignal)
