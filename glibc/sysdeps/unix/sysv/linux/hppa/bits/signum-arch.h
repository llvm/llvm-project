/* Signal number definitions.  Linux/HPPA version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _BITS_SIGNUM_ARCH_H
#define _BITS_SIGNUM_ARCH_H 1

#ifndef _SIGNAL_H
#error "Never include <bits/signum-arch.h> directly; use <signal.h> instead."
#endif

/* Adjustments and additions to the signal number constants for
   Linux/HPPA.  These values were originally chosen for HP/UX
   compatibility, but were renumbered as of kernel 3.17 and glibc 2.21
   to accommodate software (notably systemd) that assumed at least 29
   real-time signal numbers would be available.  SIGEMT and SIGLOST
   were removed, and the values of SIGSTKFLT, SIGXCPU, XIGXFSZ, and
   SIGSYS were changed, enabling __SIGRTMIN to be 32.  */

#define SIGSTKFLT	 7	/* Stack fault (obsolete).  */
#define SIGPWR		19	/* Power failure imminent.  */

/* Historical signals specified by POSIX. */
#define SIGBUS		10	/* Bus error.  */
#define SIGSYS		31	/* Bad system call.  */

/* New(er) POSIX signals (1003.1-2008, 1003.1-2013).  */
#define SIGURG		29	/* Urgent data is available at a socket.  */
#define SIGSTOP		24	/* Stop, unblockable.  */
#define SIGTSTP		25	/* Keyboard stop.  */
#define SIGCONT		26	/* Continue.  */
#define SIGCHLD		18	/* Child terminated or stopped.  */
#define SIGTTIN		27	/* Background read from control terminal.  */
#define SIGTTOU		28	/* Background write to control terminal.  */
#define SIGPOLL		22	/* Pollable event occurred (System V).  */
#define SIGXCPU		12	/* CPU time limit exceeded.  */
#define SIGVTALRM	20	/* Virtual timer expired.  */
#define SIGPROF		21	/* Profiling timer expired.  */
#define SIGXFSZ		30	/* File size limit exceeded.  */
#define SIGUSR1		16	/* User-defined signal 1.  */
#define SIGUSR2		17	/* User-defined signal 2.  */

/* Nonstandard signals found in all modern POSIX systems
   (including both BSD and Linux).  */
#define SIGWINCH	23	/* Window size change (4.3 BSD, Sun).  */

/* Archaic names for compatibility.  */
#define SIGIO		SIGPOLL	/* I/O now possible (4.2 BSD).  */
#define SIGIOT		SIGABRT	/* IOT instruction, abort() on a PDP-11.  */
#define SIGCLD		SIGCHLD	/* Old System V name */

#define __SIGRTMIN	32
#define __SIGRTMAX	64

#endif	/* <signal.h> included.  */
