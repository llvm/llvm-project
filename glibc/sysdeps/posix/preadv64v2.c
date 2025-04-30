/* Generic version of preadv2.
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

ssize_t
preadv64v2 (int fd, const struct iovec *vector, int count, off64_t offset,
	    int flags)
{
  if (flags != 0)
    {
      __set_errno (ENOTSUP);
      return -1;
    }

  if (offset == -1)
    return __readv (fd, vector, count);
  else
    return preadv64 (fd, vector, count, offset);
}

#ifdef __OFF_T_MATCHES_OFF64_T
strong_alias (preadv64v2, preadv2)
#endif
