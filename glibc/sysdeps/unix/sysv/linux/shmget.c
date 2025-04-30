/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, August 1995.

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

#include <sys/msg.h>
#include <stddef.h>
#include <ipc_priv.h>
#include <sysdep.h>

/* Return an identifier for an shared memory segment of at least size SIZE
   which is associated with KEY.  */

__attribute__((noinline))
int _shmget_internal(key_t key, size_t size, int shmflg)
{
#ifdef __ASSUME_DIRECT_SYSVIPC_SYSCALLS
  return INLINE_SYSCALL_CALL (shmget, key, size, shmflg, NULL);
#else
  return INLINE_SYSCALL_CALL (ipc, IPCOP_shmget, key, size, shmflg, NULL);
#endif
}

__attribute__((noinline))
int __shmget_nextsilicon(key_t key, size_t size, int shmflg)
{
  return _shmget_internal(key, size, shmflg);
}

__attribute__((noinline))
int shmget (key_t key, size_t size, int shmflg)
{
  return __shmget_nextsilicon(key, size, shmflg);
}