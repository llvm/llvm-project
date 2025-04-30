/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <hurd/paths.h>
#include <fcntl.h>
#include <string.h>

/* Read the contents of the symbolic link FILE_NAME into no more than
   LEN bytes of BUF.  The contents are not null-terminated.
   Returns the number of characters read, or -1 for errors.  */
ssize_t
__readlink (const char *file_name, char *buf, size_t len)
{
  error_t err;
  file_t file;
  struct stat64 st;

  file = __file_name_lookup (file_name, O_READ | O_NOLINK, 0);
  if (file == MACH_PORT_NULL)
    return -1;

  err = __io_stat (file, &st);
  if (! err)
    if (S_ISLNK (st.st_mode))
      {
	char *rbuf = buf;

	err = __io_read (file, &rbuf, &len, 0, len);
	if (!err && rbuf != buf)
	  {
	    memcpy (buf, rbuf, len);
	    __vm_deallocate (__mach_task_self (), (vm_address_t)rbuf, len);
	  }
      }
    else
      err = EINVAL;

  __mach_port_deallocate (__mach_task_self (), file);

  if (err)
    return __hurd_fail (err);
  else
    return len;
}
weak_alias (__readlink, readlink)
