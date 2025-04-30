/* Generic version of pwritev2.
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

#include <errno.h>
#include <unistd.h>
#include <sys/uio.h>

#ifndef __OFF_T_MATCHES_OFF64_T

/* Since we define no flags for pwritev2 just route to pwritev.  */
ssize_t
pwritev2 (int fd, const struct iovec *vector, int count, off_t offset,
	  int flags)
{
  if (flags != 0)
    {
      __set_errno (ENOTSUP);
      return -1;
    }

  if (offset == -1)
    return __writev (fd, vector, count);
  else
    return pwritev (fd, vector, count, offset);
}

#endif
