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
#include <sysdep-cancel.h>

ssize_t
__libc_msgrcv (int msqid, void *msgp, size_t msgsz, long int msgtyp,
	       int msgflg)
{
#ifdef __ASSUME_DIRECT_SYSVIPC_SYSCALLS
  return SYSCALL_CANCEL (msgrcv, msqid, msgp, msgsz, msgtyp, msgflg);
#else
  return SYSCALL_CANCEL (ipc, IPCOP_msgrcv, msqid, msgsz, msgflg,
			 MSGRCV_ARGS (msgp, msgtyp));
#endif
}
weak_alias (__libc_msgrcv, msgrcv)
