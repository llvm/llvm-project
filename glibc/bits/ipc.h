/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#ifndef _SYS_IPC_H
# error "Never use <bits/ipc.h> directly; include <sys/ipc.h> instead."
#endif

#include <bits/types.h>

/* Mode bits for `msgget', `semget', and `shmget'.  */
#define IPC_CREAT	01000		/* create key if key does not exist */
#define IPC_EXCL	02000		/* fail if key exists */
#define IPC_NOWAIT	04000		/* return error on wait */

/* Control commands for `msgctl', `semctl', and `shmctl'.  */
#define IPC_RMID	0		/* remove identifier */
#define IPC_SET		1		/* set `ipc_perm' options */
#define IPC_STAT	2		/* get `ipc_perm' options */

/* Special key values.  */
#define IPC_PRIVATE	((key_t) 0)	/* private key */


/* Data structure used to pass permission information to IPC operations.  */
struct ipc_perm
  {
    __uid_t uid;			/* owner's user ID */
    __gid_t gid;			/* owner's group ID */
    __uid_t cuid;			/* creator's user ID */
    __gid_t cgid;			/* creator's group ID */
    __mode_t mode;			/* read/write permission */
  };
