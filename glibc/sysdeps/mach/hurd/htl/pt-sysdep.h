/* Internal definitions for pthreads library.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _PT_SYSDEP_H
#define _PT_SYSDEP_H	1

#include <mach.h>

/* XXX */
#define _POSIX_THREAD_THREADS_MAX	64

/* The default stack size.  */
#define PTHREAD_STACK_DEFAULT	(8 * 1024 * 1024)

#define PTHREAD_SYSDEP_MEMBERS \
  thread_t kernel_thread;      \
  mach_msg_header_t wakeupmsg;

extern __thread struct __pthread *___pthread_self;
#ifdef DEBUG
#define _pthread_self()                                            \
	({                                                         \
	  struct __pthread *thread;                                \
	                                                           \
	  assert (__pthread_threads);                              \
	  thread = ___pthread_self;                                \
	                                                           \
	  assert (thread);                                         \
	  assert (({ mach_port_t ktid = __mach_thread_self ();     \
                     int ok = thread->kernel_thread == ktid;       \
                     __mach_port_deallocate (__mach_task_self (), ktid);\
                     ok; }));                                      \
          thread;                                                  \
         })
#else
#define _pthread_self() ___pthread_self
#endif

extern inline void
__attribute__ ((__always_inline__))
__pthread_stack_dealloc (void *stackaddr, size_t stacksize)
{
  __vm_deallocate (__mach_task_self (), (vm_offset_t) stackaddr, stacksize);
}

/* Change thread THREAD's program counter to PC if SET_PC is true,
   its stack pointer to SP if SET_IP is true, and its thread pointer
   to TP if SET_TP is true.  */
extern int __thread_set_pcsptp (thread_t thread,
				int set_pc, void *pc,
				int set_sp, void *sp, int set_tp, void *tp);


#endif /* pt-sysdep.h */
