/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <hurd.h>
#include <hurd/port.h>
#include <lowlevellock.h>

/* Set the process group ID of the process matching PID to PGID.
   If PID is zero, the current process's process group ID is set.
   If PGID is zero, the process ID of the process is used.  */
int
__setpgid (pid_t pid, pid_t pgid)
{
  error_t err;
  unsigned int stamp;

  stamp = _hurd_pids_changed_stamp; /* Atomic fetch.  */

  if (err = __USEPORT (PROC, __proc_setpgrp (port, pid, pgid)))
    return __hurd_fail (err);

  if (pid == 0 || pid == _hurd_pid)
    /* Synchronize with the signal thread to make sure we have
       received and processed proc_newids before returning to the user.  */
    while (_hurd_pids_changed_stamp == stamp)
      lll_wait (_hurd_pids_changed_stamp, stamp, 0);

  return 0;

}
libc_hidden_def (__setpgid)
weak_alias (__setpgid, setpgid)
