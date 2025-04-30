/* Old SysV permission definition for Linux.  Default version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <sys/ipc.h>  /* For __key_t  */
#include <kernel-features.h>

#ifdef __ASSUME_SYSVIPC_DEFAULT_IPC_64
# define __IPC_64      0x0
#else
# define __IPC_64      0x100
#endif

#ifndef __OLD_IPC_ID_TYPE
# define __OLD_IPC_ID_TYPE unsigned short int
#endif
#ifndef __OLD_IPC_MODE_TYPE
# define __OLD_IPC_MODE_TYPE unsigned short int
#endif

struct __old_ipc_perm
{
  __key_t __key;			/* Key.  */
  __OLD_IPC_ID_TYPE uid;		/* Owner's user ID.  */
  __OLD_IPC_ID_TYPE gid;		/* Owner's group ID.  */
  __OLD_IPC_ID_TYPE cuid;		/* Creator's user ID.  */
  __OLD_IPC_ID_TYPE cgid;		/* Creator's group ID.  */
  __OLD_IPC_MODE_TYPE mode;		/* Read/write permission.  */
  unsigned short int __seq;		/* Sequence number.  */
};

#define SEMCTL_ARG_ADDRESS(__arg) &__arg.array

#define MSGRCV_ARGS(__msgp, __msgtyp) \
  ((long int []){ (long int) __msgp, __msgtyp })

/* This macro is required to handle the s390 variants, which passes the
   arguments in a different order than default.  */
#define SEMTIMEDOP_IPC_ARGS(__nsops, __sops, __timeout) \
  (__nsops), 0, (__sops), (__timeout)

/* Linux SysV ipc does not provide new syscalls for 64-bit time support on
   32-bit architectures, but rather split the timestamp into high and low;
   storing the high value in previously unused fields.  */
#if (__WORDSIZE == 32 \
     && (!defined __SYSCALL_WORDSIZE || __SYSCALL_WORDSIZE == 32))
# define __IPC_TIME64 1
#else
# define __IPC_TIME64 0
#endif

#include <ipc_ops.h>
