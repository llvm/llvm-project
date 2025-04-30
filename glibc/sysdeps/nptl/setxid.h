/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <pthreadP.h>
#include <sys/single_threaded.h>
#include <sysdep.h>

#define __SETXID_1(cmd, arg1) \
  cmd.id[0] = (long int) arg1
#define __SETXID_2(cmd, arg1, arg2) \
  __SETXID_1 (cmd, arg1); cmd.id[1] = (long int) arg2
#define __SETXID_3(cmd, arg1, arg2, arg3) \
  __SETXID_2 (cmd, arg1, arg2); cmd.id[2] = (long int) arg3

#define INLINE_SETXID_SYSCALL(name, nr, args...) \
  ({									\
    int __result;							\
    if (!__libc_single_threaded)					\
      {									\
	struct xid_command __cmd;					\
	__cmd.syscall_no = __NR_##name;					\
	__SETXID_##nr (__cmd, args);					\
	__result =__nptl_setxid (&__cmd);				\
      }									\
    else								\
      __result = INLINE_SYSCALL (name, nr, args);			\
    __result;								\
   })
