/* The wrapper of clone3.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _CLONE3_H
#define _CLONE3_H	1

#include <features.h>
#include <stddef.h>
#include <bits/types.h>

__BEGIN_DECLS

/* The unsigned 64-bit and 8-byte aligned integer type.  */
typedef __U64_TYPE __aligned_uint64_t __attribute__ ((__aligned__ (8)));

/* This struct should only be used in an argument to the clone3 system
   call (along with its size argument).  It may be extended with new
   fields in the future.  */

struct clone_args
{
  /* Flags bit mask.  */
  __aligned_uint64_t flags;
  /* Where to store PID file descriptor (pid_t *).  */
  __aligned_uint64_t pidfd;
  /* Where to store child TID, in child's memory (pid_t *).  */
  __aligned_uint64_t child_tid;
  /* Where to store child TID, in parent's memory (int *). */
  __aligned_uint64_t parent_tid;
  /* Signal to deliver to parent on child termination */
  __aligned_uint64_t exit_signal;
  /* The lowest address of stack.  */
  __aligned_uint64_t stack;
  /* Size of stack.  */
  __aligned_uint64_t stack_size;
  /* Location of new TLS.  */
  __aligned_uint64_t tls;
  /* Pointer to a pid_t array (since Linux 5.5).  */
  __aligned_uint64_t set_tid;
  /* Number of elements in set_tid (since Linux 5.5). */
  __aligned_uint64_t set_tid_size;
  /* File descriptor for target cgroup of child (since Linux 5.7).  */
  __aligned_uint64_t cgroup;
};

/* The wrapper of clone3.  */
extern int clone3 (struct clone_args *__cl_args, size_t __size,
		   int (*__func) (void *__arg), void *__arg);

__END_DECLS

#endif /* clone3.h */
