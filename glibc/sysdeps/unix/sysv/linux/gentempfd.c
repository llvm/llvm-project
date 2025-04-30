/* Generate a temporary file descriptor.  Linux version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

int
__gen_tempfd (int flags)
{
  int fd = __open (P_tmpdir, O_RDWR | O_TMPFILE | O_EXCL | flags,
		   S_IRUSR | S_IWUSR);
  if (fd < 0 && errno == ENOENT && strcmp (P_tmpdir, "/tmp") != 0)
    fd = __open ("/tmp", O_RDWR | O_TMPFILE | O_EXCL | flags,
		 S_IRUSR | S_IWUSR);

  return fd;
}
libc_hidden_def (__gen_tempfd)
