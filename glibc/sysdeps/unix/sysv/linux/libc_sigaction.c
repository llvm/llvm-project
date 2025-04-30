/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <string.h>

#include <sysdep.h>
#include <sys/syscall.h>

/* New ports should not define the obsolete SA_RESTORER, however some
   architecture requires for compat mode and/or due old ABI.  */
#include <kernel_sigaction.h>

#ifndef SA_RESTORER
# define SET_SA_RESTORER(kact, act)
# define RESET_SA_RESTORER(act, kact)
#endif

/* SPARC passes the restore function as an argument to rt_sigaction.  */
#ifndef STUB
# define STUB(act, sigsetsize) (sigsetsize)
#endif

/* If ACT is not NULL, change the action for SIG to *ACT.
   If OACT is not NULL, put the old action for SIG in *OACT.  */
int
__libc_sigaction (int sig, const struct sigaction *act, struct sigaction *oact)
{
  int result;

  struct kernel_sigaction kact, koact;

  if (act)
    {
      kact.k_sa_handler = act->sa_handler;
      memcpy (&kact.sa_mask, &act->sa_mask, sizeof (sigset_t));
      kact.sa_flags = act->sa_flags;
      SET_SA_RESTORER (&kact, act);
    }

  /* XXX The size argument hopefully will have to be changed to the
     real size of the user-level sigset_t.  */
  result = INLINE_SYSCALL_CALL (rt_sigaction, sig,
				act ? &kact : NULL,
				oact ? &koact : NULL, STUB (act,
							    __NSIG_BYTES));

  if (oact && result >= 0)
    {
      oact->sa_handler = koact.k_sa_handler;
      memcpy (&oact->sa_mask, &koact.sa_mask, sizeof (sigset_t));
      oact->sa_flags = koact.sa_flags;
      RESET_SA_RESTORER (oact, &koact);
    }
  return result;
}
libc_hidden_def (__libc_sigaction)
