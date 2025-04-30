/* Implementation of the _dl_write function.  Generic version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <ldsodefs.h>
#include <libc-lock.h>
#include <sys/uio.h>

ssize_t
_dl_write (int fd, const void *buffer, size_t length)
{
  struct iovec iov = { .iov_base = (void *) buffer, .iov_len = length };
  ssize_t ret;

#if RTLD_PRIVATE_ERRNO
  /* We have to take this lock just to be sure we don't clobber the private
     errno when it's being used by another thread that cares about it.
     Yet we must be sure not to try calling the lock functions before
     the thread library is fully initialized.  */
  if (__glibc_unlikely (_dl_starting_up))
    {
      ret = __writev (fd, &iov, 1);
      if (ret < 0)
        ret = -errno;
    }
  else
    {
      __rtld_lock_lock_recursive (GL(dl_load_lock));
      ret = __writev (fd, &iov, 1);
      if (ret < 0)
        ret = -errno;
      __rtld_lock_unlock_recursive (GL(dl_load_lock));
    }
#else
  ret = __writev (fd, &iov, 1);
  if (ret < 0)
    ret = -errno;
#endif

  return ret;
}
