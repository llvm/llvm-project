/* Read data into multiple buffers.  Base implementation for preadv
   and preadv64.
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

/* Read data from file descriptor FD at the given position OFFSET
   without change the file pointer, and put the result in the buffers
   described by VECTOR, which is a vector of COUNT 'struct iovec's.
   The buffers are filled in the order specified.  Operates just like
   'pread' (see <unistd.h>) except that data are put in VECTOR instead
   of a contiguous buffer.  */
ssize_t
PREADV (int fd, const struct iovec *vector, int count, OFF_T offset)
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

  ssize_t bytes_read = PREAD (fd, buffer, bytes, offset);
  if (bytes_read < 0)
    goto end;

  /* Copy the data from BUFFER into the memory specified by VECTOR.  */
  bytes = bytes_read;
  void *buf = buffer;
  for (int i = 0; i < count; ++i)
    {
      size_t copy = MIN (vector[i].iov_len, bytes);

      memcpy (vector[i].iov_base, buf, copy);

      buf += copy;
      bytes -= copy;
      if (bytes == 0)
	break;
    }

end:
  __munmap (buffer, bytes);
  return bytes_read;
}
