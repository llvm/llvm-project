/* getsid -- Return session ID of a process.  Hurd version.
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

#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <hurd.h>

pid_t
getsid (pid_t pid)
{
  error_t err;
  pid_t sid;

  if (pid == 0)
    pid = _hurd_pid;

  err = __USEPORT (PROC, __proc_getsid (port, pid, &sid));
  if (err)
    return (pid_t) __hurd_fail (err);
  return sid;
}
libc_hidden_def (getsid)
