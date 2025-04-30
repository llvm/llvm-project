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
__setresgid (gid_t rgid, gid_t egid, gid_t sgid)
{
#ifdef __NR_setresgid32
  return INLINE_SETXID_SYSCALL (setresgid32, 3, rgid, egid, sgid);
#else
  return INLINE_SETXID_SYSCALL (setresgid, 3, rgid, egid, sgid);
#endif
}
libc_hidden_def (__setresgid)
#ifndef __setresgid
weak_alias (__setresgid, setresgid)
#endif
