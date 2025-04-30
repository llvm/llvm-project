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

/* Attach the shared memory segment associated with SHMID to the data
   segment of the calling process.  SHMADDR and SHMFLG determine how
   and where the segment is attached.  */

__attribute__((noinline))
void *
_shmat_internal(int shmid, const void *shmaddr, int shmflg)
{
#ifdef __ASSUME_DIRECT_SYSVIPC_SYSCALLS
  return (void*) INLINE_SYSCALL_CALL (shmat, shmid, shmaddr, shmflg);
#else
  unsigned long resultvar;
  void *raddr;

  resultvar = INTERNAL_SYSCALL_CALL (ipc, IPCOP_shmat, shmid, shmflg,
				     &raddr, shmaddr);
  if (INTERNAL_SYSCALL_ERROR_P (resultvar))
    return (void *) INLINE_SYSCALL_ERROR_RETURN_VALUE (INTERNAL_SYSCALL_ERRNO (resultvar));

  return raddr;
#endif
}

__attribute__((noinline))
void *
__shmat_nextsilicon(int shmid, const void *shmaddr, int shmflg)
{
  return _shmat_internal(shmid, shmaddr, shmflg);
}

__attribute__((noinline))
void *
shmat (int shmid, const void *shmaddr, int shmflg)
{
  return __shmat_nextsilicon(shmid, shmaddr, shmflg);
}