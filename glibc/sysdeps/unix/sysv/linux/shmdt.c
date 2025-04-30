/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, August 1995.

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

#include <ipc_priv.h>
#include <sysdep.h>
#include <errno.h>

/* Detach shared memory segment starting at address specified by SHMADDR
   from the caller's data segment.  */

__attribute__((noinline))
int
_shmdt_internal(const void *shmaddr)
{
#ifdef __ASSUME_DIRECT_SYSVIPC_SYSCALLS
  return INLINE_SYSCALL_CALL (shmdt, shmaddr);
#else
  return INLINE_SYSCALL_CALL (ipc, IPCOP_shmdt, 0, 0, 0, shmaddr);
#endif
}

__attribute__((noinline))
int
__shmdt_nextsilicon(const void *shmaddr)
{
  return _shmdt_internal(shmaddr);
}

__attribute__((noinline))
int
shmdt (const void *shmaddr)
{
  return __shmdt_nextsilicon(shmaddr);
}