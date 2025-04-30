/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Chris Metcalf <cmetcalf@tilera.com>, 2011.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sys/sendfile.h>
#include <stddef.h>
#include <errno.h>
#include <sysdep.h>

/* Send COUNT bytes from file associated with IN_FD starting at OFFSET to
   descriptor OUT_FD.  */
ssize_t
sendfile (int out_fd, int in_fd, off_t *offset, size_t count)
{
  __off64_t off64;
  int rc;

  if (offset != NULL)
    {
      if (*offset < 0 || (off_t) (*offset + count) < 0)
        {
          __set_errno (EINVAL);
          return -1;
        }
      off64 = *offset;
    }

  rc = INLINE_SYSCALL (sendfile64, 4, out_fd, in_fd,
                       offset ? &off64 : NULL, count);
  if (offset)
    *offset = off64;
  return rc;
}
