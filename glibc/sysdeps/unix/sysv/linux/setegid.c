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
setegid (gid_t gid)
{
  int result;

  if (gid == (gid_t) ~0)
    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);

#ifdef __NR_setresgid32
  result = INLINE_SETXID_SYSCALL (setresgid32, 3, -1, gid, -1);
#else
  result = INLINE_SETXID_SYSCALL (setresgid, 3, -1, gid, -1);
#endif

  return result;
}
#ifndef setegid
libc_hidden_def (setegid)
#endif
