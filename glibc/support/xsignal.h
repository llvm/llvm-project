/* Support functionality for using signals.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_SIGNAL_H
#define SUPPORT_SIGNAL_H

#include <signal.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/* The following functions call the corresponding libc functions and
   terminate the process on error.  */

void xraise (int sig);
sighandler_t xsignal (int sig, sighandler_t handler);
void xsigaction (int sig, const struct sigaction *newact,
                 struct sigaction *oldact);

/* The following functions call the corresponding libpthread functions
   and terminate the process on error.  */

void xpthread_sigmask (int how, const sigset_t *set, sigset_t *oldset);

/* Allocate and activate an alternate signal stack.  This stack will
   have SIZE + MINSIGSTKSZ bytes of space, rounded up to a whole
   number of pages.  There will be large (at least 1 MiB) inaccessible
   guard bands on either side of it.  The return value is a cookie
   that can be passed to xfree_sigstack to deactivate and deallocate
   the stack again.  It is not necessary to call sigaltstack after
   calling this function.  Terminates the process on error.  */
void *xalloc_sigstack (size_t size);

/* Deactivate and deallocate a signal stack created by xalloc_sigstack.  */
void xfree_sigstack (void *stack);

/* Extract the actual address and size of the alternate signal stack from
   the cookie returned by xalloc_sigstack.  */
void xget_sigstack_location (const void *stack, unsigned char **addrp,
                             size_t *sizep);

__END_DECLS

#endif /* SUPPORT_SIGNAL_H */
