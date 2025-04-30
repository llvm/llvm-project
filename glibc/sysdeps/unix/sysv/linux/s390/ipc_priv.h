/* Arch-specific SysV IPC definitions for Linux.  s390 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <sysdeps/unix/sysv/linux/ipc_priv.h>

/* The s390 sys_ipc variant has only five parameters instead of six
   (as for default variant).  The difference is the handling of
   SEMTIMEDOP where on s390 the third parameter is used as a pointer
   to a struct timespec where the generic variant uses fifth parameter.  */
#undef SEMTIMEDOP_IPC_ARGS
#define SEMTIMEDOP_IPC_ARGS(__nsops, __sops, __timeout) \
  (__nsops), (__timeout), (__sops)

#include <ipc_ops.h>
