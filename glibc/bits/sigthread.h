/* Signal handling function for threaded programs.  Generic version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#ifndef _BITS_SIGTHREAD_H
#define _BITS_SIGTHREAD_H	1

#if !defined _SIGNAL_H && !defined _PTHREAD_H
# error "Never include this file directly.  Use <signal.h> instead"
#endif

/* Modify the signal mask for the calling thread.  The arguments have the
   same meaning as for sigprocmask; in fact, this and sigprocmask might be
   the same function.  We declare this the same on all platforms, since it
   doesn't use any thread-related types.  */
extern int pthread_sigmask (int __how, const __sigset_t *__newmask,
			    __sigset_t *__oldmask) __THROW;


#endif	/* bits/sigthread.h */
