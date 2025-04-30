/* Linux implementation of preadv2.
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

#include <sys/uio.h>
#include <sysdep-cancel.h>

#ifndef __OFF_T_MATCHES_OFF64_T

ssize_t
preadv2 (int fd, const struct iovec *vector, int count, off_t offset,
	 int flags)
{
  ssize_t result = SYSCALL_CANCEL (preadv2, fd, vector, count,
				   LO_HI_LONG (offset), flags);
  if (result >= 0 || errno != ENOSYS)
    return result;

  /* Trying to emulate the preadv2 syscall flags is troublesome:

     * We can not temporary change the file state of the O_DSYNC and O_SYNC
       flags to emulate RWF_{D}SYNC (attempts to change the state of using
       fcntl are silently ignored).

     * IOCB_HIPRI requires the file opened in O_DIRECT and uses an internal
       semantic not provided by any other flag (O_NONBLOCK for instance).  */

  if (flags != 0)
    {
      __set_errno (ENOTSUP);
      return -1;
    }
  if (offset == -1)
    return __readv (fd, vector, count);
  else
    return preadv (fd, vector, count, offset);
}

#endif
