/* sendfile -- copy data directly from one file descriptor to another
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <sys/sendfile.h>
#include <stddef.h>

/* Send COUNT bytes from file associated with IN_FD starting at OFFSET to
   descriptor OUT_FD.  */
ssize_t
sendfile (int out_fd, int in_fd, off_t *offset, size_t count)
{
  if (offset == NULL || sizeof (off_t) == sizeof (off64_t))
    return __sendfile64 (out_fd, in_fd, (off64_t *) offset, count);
  else
    {
      off64_t ofs = *offset;
      ssize_t ret = __sendfile64 (out_fd, in_fd, &ofs, count);
      *offset = ofs;
      return ret;
    }
}
