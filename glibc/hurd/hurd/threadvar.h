/* Internal per-thread variables for the Hurd.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#ifndef _HURD_THREADVAR_H
#define	_HURD_THREADVAR_H

#include <features.h>
#include <tls.h>

/* The per-thread variables are found by ANDing this mask
   with the value of the stack pointer and then adding this offset.

   In the multi-threaded case, cthreads initialization sets
   __hurd_threadvar_stack_mask to ~(cthread_stack_size - 1), a mask which
   finds the base of the fixed-size cthreads stack; and
   __hurd_threadvar_stack_offset to a small offset that skips the data
   cthreads itself maintains at the base of each thread's stack.

   In the single-threaded or libpthread case, __hurd_threadvar_stack_mask is
   zero, so the stack pointer is ignored. */

extern unsigned long int __hurd_threadvar_stack_mask;
extern unsigned long int __hurd_threadvar_stack_offset;

/* The variables __hurd_sigthread_stack_base and
   __hurd_sigthread_stack_end define the bounds of the stack used by the
   signal thread, so that thread can always be specifically identified.  */

extern unsigned long int __hurd_sigthread_stack_base;
extern unsigned long int __hurd_sigthread_stack_end;

/* Store the MiG reply port reply port until we enable TLS.  */
extern mach_port_t __hurd_reply_port0;

/* This returns either the TLS reply port variable, or a single-thread variable
   when TLS is not initialized yet.  */
#define __hurd_local_reply_port (*(__LIBC_NO_TLS () ? &__hurd_reply_port0 : &THREAD_SELF->reply_port))

#endif	/* hurd/threadvar.h */
