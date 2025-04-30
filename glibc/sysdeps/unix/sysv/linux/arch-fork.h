/* arch_fork definition for Linux fork implementation.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef __ARCH_FORK_H
#define __ARCH_FORK_H

#include <sysdep.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>

/* Call the clone syscall with fork semantic.  The CTID address is used
   to store the child thread ID at its locationm, to erase it in child memory
   when the child exits, and do a wakeup on the futex at that address.

   The architecture with non-default kernel abi semantic should correctlly
   override it with one of the supported calling convention (check generic
   kernel-features.h for the clone abi variants).  */
static inline pid_t
arch_fork (void *ctid)
{
  const int flags = CLONE_CHILD_SETTID | CLONE_CHILD_CLEARTID | SIGCHLD;
  long int ret;
#ifdef __ASSUME_CLONE_BACKWARDS
# ifdef INLINE_CLONE_SYSCALL
  ret = INLINE_CLONE_SYSCALL (flags, 0, NULL, 0, ctid);
# else
  ret = INLINE_SYSCALL_CALL (clone, flags, 0, NULL, 0, ctid);
# endif
#elif defined(__ASSUME_CLONE_BACKWARDS2)
  ret = INLINE_SYSCALL_CALL (clone, 0, flags, NULL, ctid, 0);
#elif defined(__ASSUME_CLONE_BACKWARDS3)
  ret = INLINE_SYSCALL_CALL (clone, flags, 0, 0, NULL, ctid, 0);
#elif defined(__ASSUME_CLONE2)
  ret = INLINE_SYSCALL_CALL (clone2, flags, 0, 0, NULL, ctid, 0);
#elif defined(__ASSUME_CLONE_DEFAULT)
  ret = INLINE_SYSCALL_CALL (clone, flags, 0, NULL, ctid, 0);
#else
# error "Undefined clone variant"
#endif
  return ret;
}

#endif /* __ARCH_FORK_H  */
