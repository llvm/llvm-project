/* Old SysV permission definition for Linux.  x86_64 version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#define __OLD_IPC_ID_TYPE   unsigned int
#define __OLD_IPC_MODE_TYPE unsigned int
#include <sysdeps/unix/sysv/linux/ipc_priv.h>

/* SPARC semctl multiplex syscall expects the union pointed address, not
   the union address itself.  */
#undef SEMCTL_ARG_ADDRESS
#define SEMCTL_ARG_ADDRESS(__arg) __arg.array

/* Also for msgrcv it does not use the kludge on final 2 arguments.  */
#undef MSGRCV_ARGS
#define MSGRCV_ARGS(__msgp, __msgtyp) __msgp, __msgtyp

#define SEMTIMEDOP_IPC_ARGS(__nsops, __sops, __timeout) \
  (__nsops), 0, (__sops), (__timeout)

#include <ipc_ops.h>
