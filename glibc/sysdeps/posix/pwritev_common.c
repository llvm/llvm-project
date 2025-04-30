/* Write data into multiple buffers.  Base implementation for pwritev
   and pwritev64.
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

#include <unistd.h>
#include <sys/uio.h>
#include <sys/param.h>
#include <errno.h>
#include <malloc.h>

#include <ldsodefs.h>
#include <libc-pointer-arith.h>

/* Write data pointed by the buffers described by IOVEC, which is a
   vector of COUNT 'struct iovec's, to file descriptor FD at the given
   position OFFSET without change the file pointer.  The data is
   written in the order specified.  Operates just like 'write' (see
   <unistd.h>) except that the data are taken from IOVEC instead of a
   contiguous buffer.  */
ssize_t
PWRITEV (int fd, const struct iovec *vector, int count, OFF_T offset)
{
  /* Find the total number of bytes to be read.  */
  size_t bytes = 0;
  for (int i = 0; i < count; ++i)
    {
      /* Check for ssize_t overflow.  */
      if (SSIZE_MAX - bytes < vector[i].iov_len)
	{
	  __set_errno (EINVAL);
	  return -1;
	}
      bytes += vector[i].iov_len;
    }

  /* Allocate a temporary buffer to hold the data.  It could be done with a
     stack allocation, but due limitations on some system (Linux with
     O_DIRECT) it aligns the buffer to pagesize.  A possible optimization
     would be querying if the syscall would impose any alignment constraint,
     but 1. it is system specific (not meant in generic implementation), and
     2. it would make the implementation more complex, and 3. it will require
     another syscall (fcntl).  */
  void *buffer = __mmap (NULL, bytes, PROT_READ | PROT_WRITE,
		         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (__glibc_unlikely (buffer == MAP_FAILED))
    return -1;

  /* Copy the data from BUFFER into the memory specified by VECTOR.  */
  char *ptr = buffer;
  for (int i = 0; i < count; ++i)
    ptr = __mempcpy ((void *) ptr, (void *) vector[i].iov_base,
		     vector[i].iov_len);

  ssize_t ret = PWRITE (fd, buffer, bytes, offset);

  __munmap (buffer, bytes);

  return ret;
}
