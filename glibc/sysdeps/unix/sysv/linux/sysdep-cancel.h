/* Single-thread optimization definitions.  Linux version.
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

#ifndef _SYSDEP_CANCEL_H
#define _SYSDEP_CANCEL_H

#include <sysdep.h>
#include <tls.h>
#include <errno.h>

/* Set cancellation mode to asynchronous.  */
extern int __pthread_enable_asynccancel (void);
libc_hidden_proto (__pthread_enable_asynccancel)
#define LIBC_CANCEL_ASYNC() __pthread_enable_asynccancel ()

/* Reset to previous cancellation mode.  */
extern void __pthread_disable_asynccancel (int oldtype);
libc_hidden_proto (__pthread_disable_asynccancel)
#define LIBC_CANCEL_RESET(oldtype) __pthread_disable_asynccancel (oldtype)

#endif
