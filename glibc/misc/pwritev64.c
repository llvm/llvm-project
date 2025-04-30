/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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
#include <sys/uio.h>

/* Write data pointed by the buffers described by VECTOR, which is a
   vector of COUNT 'struct iovec's, to file descriptor FD at the given
   position OFFSET without change the file pointer.  The data is
   written in the order specified.  Operates just like 'pwrite' (see
   <unistd.h>) except that the data are taken from VECTOR instead of a
   contiguous buffer.  */
ssize_t
pwritev64 (int fd, const struct iovec *vector, int count, off64_t offset)
{
  __set_errno (ENOSYS);
  return -1;
}

stub_warning (pwritev64)
