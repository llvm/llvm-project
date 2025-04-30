/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <unistd.h>
#include <setxid.h>


int
seteuid (uid_t uid)
{
  int result;

  if (uid == (uid_t) ~0)
    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);

#ifdef __NR_setresuid32
  result = INLINE_SETXID_SYSCALL (setresuid32, 3, -1, uid, -1);
#else
  result = INLINE_SETXID_SYSCALL (setresuid, 3, -1, uid, -1);
#endif

  return result;
}
#ifndef seteuid
libc_hidden_def (seteuid)
#endif
