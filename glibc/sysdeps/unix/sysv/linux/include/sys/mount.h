/* Internal declarations for sys/mount.h.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _INCLUDE_SYS_MOUNT_H
#define _INCLUDE_SYS_MOUNT_H	1

#include_next <sys/mount.h>

# ifndef _ISOMAC

extern __typeof (umount) __umount __THROW;
extern __typeof (umount2) __umount2 __THROW;
libc_hidden_proto (__umount2)

# endif /* _ISOMAC */
#endif /* sys/sysinfo.h */
