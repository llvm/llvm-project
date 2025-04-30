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

#ifndef _BITS_SIGACTION_H
#define _BITS_SIGACTION_H 1

#ifndef _SIGNAL_H
# error "Never include <bits/sigaction.h> directly; use <signal.h> instead."
#endif

/* These definitions match those used by the 4.4 BSD kernel.
   If the operating system has a `sigaction' system call that correctly
   implements the POSIX.1 behavior, there should be a system-dependent
   version of this file that defines `struct sigaction' and the `SA_*'
   constants appropriately.  */

/* Structure describing the action to be taken when a signal arrives.  */
struct sigaction
  {
    /* Signal handler.  */
#if defined __USE_POSIX199309 || defined __USE_XOPEN_EXTENDED
    union
      {
	/* Used if SA_SIGINFO is not set.  */
	__sighandler_t sa_handler;
	/* Used if SA_SIGINFO is set.  */
	void (*sa_sigaction) (int, siginfo_t *, void *);
      }
    __sigaction_handler;
# define sa_handler	__sigaction_handler.sa_handler
# define sa_sigaction	__sigaction_handler.sa_sigaction
#else
    __sighandler_t sa_handler;
#endif

    /* Additional set of signals to be blocked.  */
    __sigset_t sa_mask;

    /* Special flags.  */
    int sa_flags;
  };

/* Bits in `sa_flags'.  */
#if defined __USE_XOPEN_EXTENDED || defined __USE_MISC
# define SA_ONSTACK	0x0001	/* Take signal on signal stack.  */
#endif
#if defined __USE_XOPEN_EXTENDED || defined __USE_XOPEN2K8
# define SA_RESTART	0x0002	/* Restart syscall on signal return.  */
# define SA_NODEFER	0x0010	/* Don't automatically block the signal when
				    its handler is being executed.  */
# define SA_RESETHAND	0x0004	/* Reset to SIG_DFL on entry to handler.  */
#endif
#define	SA_NOCLDSTOP	0x0008	/* Don't send SIGCHLD when children stop.  */
#define SA_SIGINFO	0x0040	/* Signal handler with SA_SIGINFO args */

#ifdef __USE_MISC
# define SA_INTERRUPT	0	/* Historical no-op ("not SA_RESTART").  */

/* Some aliases for the SA_ constants.  */
# define SA_NOMASK    SA_NODEFER
# define SA_ONESHOT   SA_RESETHAND
# define SA_STACK     SA_ONSTACK
#endif


/* Values for the HOW argument to `sigprocmask'.  */
#define	SIG_BLOCK	1	/* Block signals.  */
#define	SIG_UNBLOCK	2	/* Unblock signals.  */
#define	SIG_SETMASK	3	/* Set the set of blocked signals.  */

#endif
