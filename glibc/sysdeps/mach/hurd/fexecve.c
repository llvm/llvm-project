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

#include <unistd.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <errno.h>

/* Execute the file FD refers to, overlaying the running program image.  */

int
fexecve (int fd, char *const argv[], char *const envp[])
{
  file_t file;
  error_t err;
  enum retry_type doretry;
  char retryname[1024];

  err = HURD_DPORT_USE (fd,
      __dir_lookup (port, "", O_EXEC, 0, &doretry, retryname, &file));

  if (! err && (doretry != FS_RETRY_NORMAL || retryname[0] != '\0'))
    err = EGRATUITOUS;
  if (err)
    return __hurd_fail(err);

  err = _hurd_exec_paths (__mach_task_self (), file, NULL, NULL, argv, envp);
  if (! err)
    err = EGRATUITOUS;

  __mach_port_deallocate (__mach_task_self (), file);
  return __hurd_fail (err);
}
